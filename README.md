# Unifews
This is the original code for *Unifews: Unified Entry-Wise Sparsification for Efficient Graph Neural Network*

## Dependencies
### Python
In `env.txt` and can be installed by:
```bash
conda create --name <env> --file env.txt
```

### C++
* [eigen3](https://eigen.tuxfamily.org/index.php?title=Main_Page)

## Experiment
### Data Preparation
1. Use `utils/data_transfer.py` to generate processed files under path `data/[dataset_name]` similar to the example folder `data/cora`:
  * `feats.npy`: features in .npy array
  * `labels.npz`: node label information
    * 'label': labels (number or one-hot)
    * 'idx_train/idx_val/idx_test': indices of training/validation/test nodes
  * `adj_el.bin`, `adj_pl.bin`, `attribute.txt`, `deg.npz`: graph files for precomputation

### Decoupled Model Propagation
1. Environment: CMake 3.16, C++ 14. 
2. Compile Cython:
```bash
python setup.py build_ext --inplace
```

### Model Training
1. Run full-batch experiment: 
```bash
python run_fb.py -f [seed] -c [config_file] -v [device]
```
2. Run mini-batch experiment
```bash
python run_mb.py -f [seed] -c [config_file] -v [device]
```

## Reference & Links
### Datasets
* cora, citeseer, pubmed: [Pytorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.Planetoid.html#torch_geometric.datasets.Planetoid)
* arxiv, products, papers100m: [OGBl](https://ogb.stanford.edu/docs/home/)

### Baselines
- [GLT](https://github.com/VITA-Group/Unified-LTH-GNN): *A Unified Lottery Ticket Hypothesis for Graph Neural Networks*
- [GEBT](https://github.com/GATECH-EIC/Early-Bird-GCN): *Early-Bird GCNs: Graph-Network Co-optimization towards More Efficient GCN Training and Inference via Drawing Early-Bird Lottery Tickets*
- [CGP](https://github.com/LiuChuang0059/CGP/): *Comprehensive Graph Gradual Pruning for Sparse Training in Graph Neural Networks*
- [DSpar](https://github.com/zirui-ray-liu/DSpar_tmlr): *DSpar: An Embarrassingly Simple Strategy for Efficient GNN Training and Inference via Degree-Based Sparsification*
- [NDLS](https://github.com/zwt233/NDLS): *Node Dependent Local Smoothing for Scalable Graph Learning*
- [NIGCN](https://github.com/kkhuang81/NIGCN): *Node-wise Diffusion for Scalable Graph Learning*
