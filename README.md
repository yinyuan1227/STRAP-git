
# Scalable Graph Embeddings via Sparse Transpose Proximities


## Tested Environment
- Ubuntu
- C++ 11
- GCC 4.8


## Preparation
Place the prepared data [file].txt in the [NR_Dataset](https://github.com/yinyuan1227/STRAP-git/tree/master/NR_Dataset). Note that the first row of data is the node size and each row is the information of each edge [outNode] [inNode].

Besides, directed graph and undirected graph should be distinguished. 

Datasets used in our paper are provided in [NR_Dataset](https://github.com/yinyuan1227/STRAP-git/tree/master/NR_Dataset).


## Compilations
```sh
bash compile.sh
```


## Usage
We provide two versions of the code to ensure reproducibility.
### STRAP based on SVD
We write a SVD version based on Eigen 3.X, which is a free software that can handle many different linear algebra operations and also has a geometry framework.  

```
./STRAP_SVD_U <graph_name> <data_path> <emb_path> <alpha> <iteration> <error> <threads>
```
**Parameters**

- graph_name: name of target graph
- data_path: path to load source file 
- emb_path: path to save embedding files
- alpha: parameter for *PPR*
- iteration: parameter for *SVD*
- error: parameter for *Backward Push*
- threads

**Examples**

For undirected graph:
```
./STRAP_SVD_U BlogCatalog-u NR_Dataset/ NR_EB/ 0.5 12 0.00001 24
```

For directed graph:
```
./STRAP_SVD_D wikivote NR_Dataset/ NR_EB/ 0.5 12 0.00001 24
```


### STRAP based on frPCA
***Results in our paper are all based on this version.***

In this version we make use of [frPCA](https://github.com/XuFengthucs/frPCA_sparse) to get better performance.

```
./STRAP_FRPCA_U <graph_name> <data_path> <emb_path> <alpha> <iteration> <error> <threads>
```

**Parameters**

- iteration: parameter for *frPCA*

Others are the same as above.


**Examples**

For undirected graph:
```
./STRAP_FRPCA_U BlogCatalog-u NR_Dataset/ NR_EB/ 0.5 12 0.00001 24
```

For directed graph:
```
./STRAP_FRPCA_D wikivote NR_Dataset/ NR_EB/ 0.5 12 0.00001 24
```



## Experiments
### Graph Reconstruction
Train the embeddings of a full graph and then reconstruct it. The code to calculate reconstruction precision is provided:

```
./NET_RE_U BlogCatalog-u strap_frpca_u
```

```
./NET_RE_D wikivote strap_frpca_d
```
For big graphs, like *YouTube*, we sample a subgraph to do reconstruction. 

### Link Prediction
First, split the graph into training/testing set and generate negative samples. Datasets will be saved into [LP_Dataset](https://github.com/yinyuan1227/STRAP-git/tree/master/LP_Dataset) separately. The ratio of testing part can be assigned:
 
```
./GEN_DATA_U BlogCatalog-u 0.5
```

```
./GEN_DATA_D wikivote 0.5
```
Then get embeddings of the training set. Predict missing edges via score $s_u
 \cdot t_v$. The code to calculate link prediction precision is provided:

 ```
 ./LINK_PRE_U BlogCatalog-u strap_frpca_u
 ```
 
 ```
 ./LINK_PRE_D wikivote strap_frpca_d
 ```
 
### Node Classification
Generate a classifier using the embeddings of full graph, the provided labels and the training set. The performance is evaluated in terms of average Micro-F1 and average Macro-F1.
This part is implemented in Python 3.4 and sklearn 0.20.1:

```
python labelclassification.py BlogCatalog-u strap_frpca_u
```


## Citing
Please cite our paper if you choose to use our code. 

```
```
