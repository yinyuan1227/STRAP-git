
# Scalable Graph Embeddings via Sparse Transpose Proximities


## Tested Environment
- Ubuntu
- C++ 11
- GCC 4.8
- Intel C++ Compiler
- Boost (optional, only link prediction requires this)


## Preparation
Place the prepared data [file].txt in the [NR_Dataset](https://github.com/yinyuan1227/STRAP-git/tree/master/NR_Dataset). Note that the first row of data is the node size and each row is the information of each edge [outNode] [inNode].

Besides, directed graph and undirected graph should be distinguished. 

Datasets used in our paper are provided in [NR_Dataset](https://github.com/yinyuan1227/STRAP-git/tree/master/NR_Dataset).

|Data Set|Directed|N|M|
|:------|:-----:|------:|------:|
|[BlogCatalog](http://socialcomputing.asu.edu/pages/datasets)|No|10312|333983|
|[Flickr](http://socialcomputing.asu.edu/pages/datasets)|No|80513|5899882|
|[YouTube](http://socialcomputing.asu.edu/pages/datasets)|No|1138499|2990443|
|[WikiVote](http://snap.stanford.edu/data/wiki-Vote.html)|Yes|7115|103689|
|[Slashdot](http://snap.stanford.edu/data/soc-Slashdot0902.html)|Yes|82168|870161|
|[Euro](https://github.com/leoribeiro/struc2vec/)|No|399|5993|
|[Brazil](https://github.com/leoribeiro/struc2vec/)|No|131|1003|


## Compilations
```sh
bash compile.sh
```
**Move the files in the frPCA folder to the root directory before compiling.**


## Usage
We provide two versions of the code to ensure reproducibility.
### STRAP based on SVD
We write a SVD version based on *Eigen 3.x*.  

```
./STRAP_SVD_U <graph_name> <data_path> <emb_path> <alpha> <iteration> <error> <threads>
```
**Parameters**

- graph_name: name of target graph
- data_path: path to load source file 
- emb_path: path to save embedding files
- alpha: parameter for *PPR*
- iteration: parameter for *SVD*
- error: parameter for *[Backward Push](https://arxiv.org/abs/1507.05999)*
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

In this version we make use of *[frPCA](https://arxiv.org/abs/1810.06825)* to get better performance.

```
./STRAP_FRPCA_U <graph_name> <data_path> <emb_path> <alpha> <iteration> <error> <threads>
```

**Parameters**

- iteration: parameter for *[frPCA](https://arxiv.org/abs/1810.06825)*

- others are the same as above


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
@inproceedings{10.1145/3292500.3330860,
author = {Yin, Yuan and Wei, Zhewei},
title = {Scalable Graph Embeddings via Sparse Transpose Proximities},
year = {2019},
isbn = {9781450362016},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3292500.3330860},
doi = {10.1145/3292500.3330860},
booktitle = {Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining},
pages = {1429–1437},
numpages = {9},
keywords = {network representation learning, personalized pagerank, graph embedding},
location = {Anchorage, AK, USA},
series = {KDD ’19}
}
```
