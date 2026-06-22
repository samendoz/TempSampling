#!/usr/bin/env bash
# experimental_setup.sh — reproducible TempSampling environment setup
#
# Tested stack: Python 3.12, torch 2.2.0+cu121, dgl 2.1.0+cu121, numpy 1.x
# Required system: g++ >= 7.5.0, CUDA-capable GPU
#
# Usage:
#   bash experimental_setup.sh          # creates ts_env/
#   source ts_env/bin/activate
#   python train_simple.py --extra_config ./config/adapt_exp/test_01.yml

set -e

ENV_NAME="ts_env"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITE_PACKAGES="$REPO_DIR/$ENV_NAME/lib/python3.12/site-packages"

# ---------------------------------------------------------------------------
# 1. Locate Python 3.12
# ---------------------------------------------------------------------------
PYTHON_BIN=$(command -v python3 || command -v python)
if [ -z "$PYTHON_BIN" ]; then
    echo "ERROR: No python3 found."
    exit 1
fi
echo "Using Python: $PYTHON_BIN ($($PYTHON_BIN --version 2>&1))"

# ---------------------------------------------------------------------------
# 2. Create or reuse virtual environment
# ---------------------------------------------------------------------------
if [ -f "$ENV_NAME/bin/activate" ]; then
    echo "Virtual environment '$ENV_NAME' already exists — reusing it."
else
    [ -d "$ENV_NAME" ] && rm -rf "$ENV_NAME"
    echo "Creating virtual environment '$ENV_NAME'..."
    $PYTHON_BIN -m venv "$ENV_NAME"
fi

source "$ENV_NAME/bin/activate"
echo "Activated: $(which python)"

pip install --upgrade pip --quiet

# ---------------------------------------------------------------------------
# 3. Install PyTorch 2.2.0 with CUDA 12.1 first (pinned before DGL can
#    override it via torchdata's loose >=2 dependency)
# ---------------------------------------------------------------------------
echo "Installing PyTorch 2.2.0+cu121..."
pip install torch==2.2.0 --index-url https://download.pytorch.org/whl/cu121

# ---------------------------------------------------------------------------
# 4. Install DGL 2.1.0+cu121 WITHOUT deps to prevent torchdata from pulling
#    torch 2.11 and overriding the pinned version above
# ---------------------------------------------------------------------------
echo "Installing DGL 2.1.0+cu121 (no-deps)..."
pip install dgl==2.1.0+cu121 --no-deps -f https://data.dgl.ai/wheels/cu121/repo.html

# Install DGL's remaining deps manually (omit torchdata intentionally)
pip install scipy networkx requests psutil

# ---------------------------------------------------------------------------
# 5. Core project dependencies
# ---------------------------------------------------------------------------
echo "Installing project dependencies..."
pip install \
    "numpy<2" \
    packaging \
    "pandas>=1.1.5" \
    "pyyaml>=5.4.1" \
    "tqdm>=4.61.0" \
    "pybind11>=2.6.2" \
    scikit-learn \
    setuptools \
    nvtx \
    matplotlib

# ---------------------------------------------------------------------------
# 6. torch-scatter (matches torch 2.2.0+cu121)
# ---------------------------------------------------------------------------
echo "Installing torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.2.0+cu121.html

# ---------------------------------------------------------------------------
# 7. Patch DGL: stub out graphbolt entirely (TempSampling does not use it;
#    DGL 2.x requires torchdata.datapipes and a native .so that do not exist
#    for this torch/CUDA version)
# ---------------------------------------------------------------------------
echo "Patching DGL graphbolt..."
cat > "$SITE_PACKAGES/dgl/graphbolt/__init__.py" << 'PYEOF'
# stubbed out — TempSampling does not use graphbolt
PYEOF

# ---------------------------------------------------------------------------
# 8. Stub torchdata.datapipes and torchdata.dataloader2 (imported by DGL
#    graphbolt before the stub above fully stops the chain in older DGL builds)
# ---------------------------------------------------------------------------
echo "Creating torchdata stubs..."
mkdir -p "$SITE_PACKAGES/torchdata/datapipes/iter"
mkdir -p "$SITE_PACKAGES/torchdata/dataloader2"

cat > "$SITE_PACKAGES/torchdata/datapipes/__init__.py" << 'PYEOF'
# stub
PYEOF

cat > "$SITE_PACKAGES/torchdata/datapipes/iter/__init__.py" << 'PYEOF'
class IterDataPipe:
    pass
class IterableWrapper(IterDataPipe):
    pass
class Mapper(IterDataPipe):
    pass
PYEOF

cat > "$SITE_PACKAGES/torchdata/dataloader2/__init__.py" << 'PYEOF'
# stub
PYEOF

cat > "$SITE_PACKAGES/torchdata/dataloader2/graph.py" << 'PYEOF'
# stub
PYEOF

# ---------------------------------------------------------------------------
# 9. Ensure g++ is available, then compile the C++ extensions
#    TempSampling has TWO extensions:
#      a) sampler_core  — pybind11-based temporal neighbor sampler (setup.py)
#      b) ColorSamplerCore — torch CppExtension color batch sampler (setup_sampler.py)
# ---------------------------------------------------------------------------
if ! command -v g++ &>/dev/null; then
    echo "g++ not found. Installing build-essential..."
    sudo apt-get update -qq && sudo apt-get install -y build-essential python3-dev
fi
echo "g++ found: $(g++ --version | head -1)"

echo "Compiling temporal neighbor sampler (sampler_core)..."
cd "$REPO_DIR"
python setup.py build_ext --inplace

echo "Compiling color batch sampler (ColorSamplerCore)..."
python setup_sampler.py build_ext --inplace

echo ""
echo "Setup complete."
echo "Activate with:  source $REPO_DIR/$ENV_NAME/bin/activate"
echo "Run training:   python train_simple.py --extra_config ./config/adapt_exp/test_01.yml"
