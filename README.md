## Overview

This repo is the experimental code for *Cascade: A Dependency-aware Efficient Training Framework for Temporal Graph Neural Network* with dependency/stable check. For open-source release, please refer to future links.

## Requirements
- python >= 3.6.13
- pytorch >= 1.8.1
- pandas >= 1.1.5
- numpy >= 1.19.5
- dgl >= 0.6.1
- pyyaml >= 5.4.1
- tqdm >= 4.61.0
- pybind11 >= 2.6.2
- g++ >= 7.5.0
- openmp >= 201511

The sampler is implemented using C++, please compile the sampler first with the following command
> chmod +x build.sh
> ./build.sh

## Dataset

You can download WIKI, REDDIT, MOOC, LASTFM, GDELT, and MAG from AWS S3 bucket using the `down.sh` script. 

To use your own dataset, you need to put the following files in the folder `\DATA\\<NameOfYourDataset>\`

1. `edges.csv`: The file that stores temporal edge informations. The csv should have the following columns with the header as `,src,dst,time,ext_roll` where each of the column refers to edge index (start with zero), source node index (start with zero), destination node index, time stamp, extrapolation roll (0 for training edges, 1 for validation edges, 2 for test edges). The CSV should be sorted by time ascendingly.
2. `ext_full.npz`: The T-CSR representation of the temporal graph. We provide a script to generate this file from `edges.csv`. You can use the following command to use the script 
    >python gen_graph.py --data \<NameOfYourDataset>
3. `edge_features.pt` (optional): The torch tensor that stores the edge featrues row-wise with shape (num edges, dim edge features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
4. `node_features.pt` (optional): The torch tensor that stores the node featrues row-wise with shape (num nodes, dim node features). *Note: at least one of `edge_features.pt` or `node_features.pt` should present.*
5. `labels.csv` (optional): The file contains node labels for dynamic node classification task. The csv should have the following columns with the header as `,node,time,label,ext_roll` where each of the column refers to node label index (start with zero), node index (start with zero), time stamp, node label, extrapolation roll (0 for training node labels, 1 for validation node labels, 2 for test node labels). The CSV should be sorted by time ascendingly.

For WIKI_TALK and STACK_OVERFLOW (optional), please download from following links:
WIKI_TALK: 'https://snap.stanford.edu/data/wiki-talk-temporal.txt.gz'
STACK_OVERFLOW: 'https://snap.stanford.edu/data/sx-stackoverflow.txt.gz'

Unzip to:
`\DATA\WIKI_TALK\` and `\DATA\STACK_OVERFLOW\`

Then run:
> python dataset_transform.py --data WIKI_TALK
> python gen_graph.py --data WIKI_TALK

and
> python dataset_transform.py --data STACK_OVERFLOW
> python gen_graph.py --data STACK_OVERFLOW


## Configuration Files

There are example configuration files for five temporal GNN methods: JODIE, DySAT, TGAT, TGN and TGAT. The configuration files for single GPU training are located at `/config/` while the multiple GPUs training configuration files are located at `/config/dist/`.

There are a few experimental configuration files for TGN/APAN/TGAT trainer experiments in five datasets: WIKI, REDDIT, WIKI-TALK, STACK_OVERFLOW and GDELT. The configuration files are located at `/config/adapt_exp/`


### Run on Single GPU Link Prediction
> python train_simple.py --extra_config \<pathToExperimentalConfigs>

or

you can run one time

> chmod +x run.sh

then customize the script in it then run

> ./run.sh

## Result Collection

### Single run with logging

Wraps `train.py` or `train_simple.py` with the same arguments. Zero training overhead — uses `tee` to capture stdout/stderr while training runs, then parses the log into a `.log` and `.csv` file under `results/`.

```bash
# adapt-exp style (train_simple.py)
bash collect_results.sh --extra_config config/adapt_exp/test_01.yml

# standard style (train.py)
bash collect_results.sh --data WIKI --config config/TGN.yml
```

### Batch runs

```bash
# run all configs in a directory
bash run_all_experiments.sh --exp_dir config/adapt_exp

# run specific configs from a directory
bash run_all_experiments.sh --exp_dir config/adapt_exp --configs "test_01 test_02"

# run specific models on a dataset (looks up config/adapt_exp/<DATA>/<MODEL>.yml)
bash run_all_experiments.sh --exp_dir config/adapt_exp/WIKI --configs "TGAT TGN APAN"

# pass extra args to train_simple.py (after --)
bash run_all_experiments.sh --exp_dir config/adapt_exp/WIKI --configs "TGAT TGN" -- --eval_neg_samples 10
```
