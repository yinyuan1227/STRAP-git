#ifndef GRAPH_H
#define GRAPH_H

#define SetBit(A, k)     ( A[(k/32)] |= (1 << (k%32)) )
#define ClearBit(A, k)   ( A[(k/32)] &= ~(1 << (k%32)) )
#define TestBit(A, k)    ( A[(k/32)] & (1 << (k%32)) )

#define DSetBit(A, k, j, n)     ( A[(k*(n/32)+(j/32))] |= (1 << (((k%32)*(n%32))%32+j%32)%32) )
#define DClearBit(A, k, j, n)   ( A[(k*(n/32)+(j/32))] &= ~(1 << (((k%32)*(n%32))%32+j%32)%32) )
#define DTestBit(A, k, j, n)    ( A[(k*(n/32)+(j/32))] & (1 << (((k%32)*(n%32))%32+j%32)%32) )


#include <cstdlib>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <time.h>
#include <vector>
#include <unordered_set>

using namespace std;


class BitMatrix {
public:
  long long int n;
  long long int narray;
  int *bitMatrix;

  BitMatrix() {
  }

  ~BitMatrix() {
  }

  void ConBitMatrix(int nodenumber) {
    n = nodenumber;
    narray = n * (n / 32) + 1;
    bitMatrix = new int[narray];
    for (int i = 0; i < narray; i++) {
      bitMatrix[i] = 0;
    }
  }

  void Update(int i, int j) {
    if (!DTestBit(bitMatrix, i, j, n)) {
      DSetBit(bitMatrix, i, j, n);
    }
  }

  int Find(int i, int j) {
    return DTestBit(bitMatrix, i, j, n);
  }
};


class Graph {
public:
  int n;    //number of nodes
  int m;    //number of edges
  int **inAdjList;
  int **outAdjList;
  int *indegree;
  int *outdegree;

  Graph() {
  }

  ~Graph() {
  }

  void inputGraph(string filename) {
    cout << filename << endl;
    clock_t t1 = clock();
    m = 0;
    ifstream infile(filename.c_str());
    infile >> n;
    cout << "n=" << n << endl;
    indegree = new int[n];
    outdegree = new int[n];
    for (int i = 0; i < n; i++) {
      indegree[i] = 0;
      outdegree[i] = 0;
    }

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
    }
    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      outdegree[from]++;
      indegree[to]++;
    }
    cout << "..." << endl;
    inAdjList = new int *[n];
    outAdjList = new int *[n];
    for (int i = 0; i < n; i++) {
      inAdjList[i] = new int[indegree[i]];
      outAdjList[i] = new int[outdegree[i]];
    }
    int *pointer_in = new int[n];
    int *pointer_out = new int[n];
    for (int i = 0; i < n; i++) {
      pointer_in[i] = 0;
      pointer_out[i] = 0;
    }


    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      outAdjList[from][pointer_out[from]] = to;
      pointer_out[from]++;
      inAdjList[to][pointer_in[to]] = from;
      pointer_in[to]++;
      m++;
    }

    for (int i = 0; i < n; i++) {
      sort(outAdjList[i], outAdjList[i] + outdegree[i]);
    }

    delete[] pointer_in;
    delete[] pointer_out;

    clock_t t2 = clock();
    cout << "m=" << m << endl;
  }

  void RandomSplitGraph(string inFilename, string outFilename1, string outFilename2, double percent) {
    clock_t t1 = clock();
    int m1 = 0;
    ifstream infile(inFilename.c_str());
    ofstream outfile1(outFilename1.c_str());
    ofstream outfile2(outFilename2.c_str());
    int n1;
    infile >> n1;
    outfile1 << n1 << endl;
    outfile2 << n1 << endl;

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
      m1++;
    }
    int m2 = percent * m1;
    for (int i = 0; i < m2; i++) {
      int r = rand() % (m1 - i);
      iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
      outfile1 << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }
    for (int i = m2; i < edge_vec.size(); i++) {
      outfile2 << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }
  }

  void NegativeSamples(string outFilename, double percent) {
    int sample_number = 0;
    int total_sample_number = percent * m;
    ofstream outfile(outFilename.c_str());
    vector<unordered_set<int>> adjM(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < outdegree[i]; j++) {
        adjM[i].insert(outAdjList[i][j]);
      }
    }
    outfile << n << endl;
    int npair = 0;
    int nsample = 0;
    while (sample_number < total_sample_number) {
      int i = rand() % n;
      int j = rand() % n;
      npair++;
      if (adjM[i].find(j) == adjM[i].end()) {
        outfile << i << " " << j << endl;
        sample_number++;
        nsample++;
      }
    }
    cout << "npair: " << npair << " nsample: " << nsample << endl;
  }

  int getInSize(int vert) {
    return indegree[vert];
  }

  int getInVert(int vert, int pos) {
    return inAdjList[vert][pos];
  }

  int getOutSize(int vert) {
    return outdegree[vert];
  }

  int getOutVert(int vert, int pos) {
    return outAdjList[vert][pos];
  }

};


class UGraph {
public:
  int n;    //number of nodes
  int m;    //number of edges
  int **AdjList;
  int *degree;

  UGraph() {
  }

  ~UGraph() {
  }

  void inputGraph(string filename) {
    cout << filename << endl;
    clock_t t1 = clock();
    m = 0;
    ifstream infile(filename.c_str());
    infile >> n;
    cout << "n=" << n << endl;
    degree = new int[n];
    for (int i = 0; i < n; i++) {
      degree[i] = 0;
    }

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
    }
    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      degree[from]++;
      degree[to]++;
    }
    cout << "..." << endl;
    AdjList = new int *[n];
    for (int i = 0; i < n; i++) {
      AdjList[i] = new int[degree[i]];
    }
    int *pointer_out = new int[n];
    for (int i = 0; i < n; i++) {
      pointer_out[i] = 0;
    }

    for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
      from = it->first;
      to = it->second;
      AdjList[from][pointer_out[from]] = to;
      pointer_out[from]++;
      AdjList[to][pointer_out[to]] = from;
      pointer_out[to]++;
      m++;
    }

    for (int i = 0; i < n; i++) {
      sort(AdjList[i], AdjList[i] + degree[i]);
    }

    delete[] pointer_out;

    clock_t t2 = clock();
    cout << "m=" << m << endl;
  }

  void RandomSplitGraph(string inFilename, string outFilename1, string outFilename2, double percent) {
    clock_t t1 = clock();
    int m1 = 0;
    ifstream infile(inFilename.c_str());
    ofstream outfile1(outFilename1.c_str());
    ofstream outfile2(outFilename2.c_str());
    int n1;
    infile >> n1;
    outfile1 << n1 << endl;
    outfile2 << n1 << endl;

    //read graph and get degree info
    vector<pair<int, int>> edge_vec;
    int from;
    int to;
    while (infile.good()) {
      infile >> from >> to;
      edge_vec.push_back(make_pair(from, to));
      m1++;
    }
    int m2 = percent * m1;
    for (int i = 0; i < m2; i++) {
      int r = rand() % (m1 - i);
      iter_swap(edge_vec.begin() + i, edge_vec.begin() + i + r);
      outfile1 << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }
    for (int i = m2; i < edge_vec.size(); i++) {
      outfile2 << edge_vec[i].first << " " << edge_vec[i].second << endl;
    }
  }

  void NegativeSamples(string outFilename, double percent) {
    int sample_number = 0;
    int total_sample_number = percent * m;
    ofstream outfile(outFilename.c_str());
    vector<unordered_set<int>> adjM(n);
    for (int i = 0; i < n; i++) {
      for (int j = 0; j < degree[i]; j++) {
        adjM[i].insert(AdjList[i][j]);
      }
    }
    outfile << n << endl;
    int npair = 0;
    int nsample = 0;
    while (sample_number < total_sample_number) {
      int i = rand() % n;
      int j = rand() % n;
      npair++;
      if (adjM[i].find(j) == adjM[i].end() && adjM[j].find(i) == adjM[j].end()) {
        outfile << i << " " << j << endl;
        sample_number++;
        nsample++;
      }
    }
    cout << "npair: " << npair << " nsample: " << nsample << endl;
  }

  int getSize(int vert) {
    return degree[vert];
  }

  int getVert(int vert, int pos) {
    return AdjList[vert][pos];
  }

};


class Node_Set {
public:
  int vert;
  int bit_vert;
  int *HashKey;
  int *HashValue;
  int KeyNumber;


  Node_Set(int n) {
    vert = n;
    bit_vert = n / 32 + 1;
    HashKey = new int[vert];
    HashValue = new int[bit_vert];
    for (int i = 0; i < vert; i++) {
      HashKey[i] = 0;
    }
    for (int i = 0; i < bit_vert; i++) {
      HashValue[i] = 0;
    }
    KeyNumber = 0;
  }


  void Push(int node) {

    if (!TestBit(HashValue, node)) {
      HashKey[KeyNumber] = node;
      KeyNumber++;
    }
    SetBit(HashValue, node);
  }


  int Pop() {
    if (KeyNumber == 0) {
      return -1;
    } else {
      int k = HashKey[KeyNumber - 1];
      ClearBit(HashValue, k);
      KeyNumber--;
      return k;
    }
  }

  void Clean() {
    for (int i = 0; i < KeyNumber; i++) {
      ClearBit(HashValue, HashKey[i]);
      HashKey[i] = 0;
    }
    KeyNumber = 0;
  }

  ~Node_Set() {
    delete[] HashKey;
    delete[] HashValue;
  }
};


#endif

