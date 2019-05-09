#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_map>
#include "Graph.h"
#include "Eigen/Dense"
#include "SVD.h"


using namespace Eigen;

template<typename M>
M load_csv(const std::string &path) {
  std::ifstream indata;
  indata.open(path);
  std::string line;
  std::vector<double> values;
  uint rows = 0;
  while (std::getline(indata, line)) {
    std::stringstream lineStream(line);
    std::string cell;
    while (std::getline(lineStream, cell, ',')) {
      values.push_back(std::stod(cell));
    }
    ++rows;
  }
  return Map<const Matrix<typename M::Scalar, M::RowsAtCompileTime, M::ColsAtCompileTime, RowMajor>>(values.data(), rows, values.size()/rows);
}


bool maxScoreCmp(const pair<double, pair<int, int>> &a, const pair<double, pair<int, int>> &b) {
  return a.first > b.first;
}

const static IOFormat CSVFormat(StreamPrecision, DontAlignCols, ", ", "\n");


int main(int argc, char **argv) {
  char *endptr;
  string queryname = argv[1];
  string methodname = argv[2];
  string inUfile = "NR_EB/" + queryname + "_" + methodname + "_U.csv";
  string inVfile = "NR_EB/" + queryname + "_" + methodname + "_V.csv";
  string dataset = "NR_Dataset/" + queryname + ".txt";
  UGraph g;
  g.inputGraph(dataset);
  MatrixXd embU = load_csv<MatrixXd>(inUfile);
  MatrixXd embV = load_csv<MatrixXd>(inVfile);
  int d = embU.cols();

  ifstream infile(dataset.c_str());
  int n = 0;
  infile >> n;
  vector <pair<int, int>> edge_vec;
  int from;
  int to;
  while (infile.good()) {
    infile >> from >> to;
    edge_vec.push_back(make_pair(from, to));
  }
  random_shuffle(edge_vec.begin(), edge_vec.end());

  // if graph size > 100000, use sampled graph, otherwise use original graph
  // sample a subgraph using random edge induction method
  srand((unsigned) time(0));
  cout << "Constructing sample graph using edge induction..." << endl;
  int sample_init_m = g.m;
  if (g.n >= 100000) {
    sample_init_m = min(100000, g.m);
  }

  //use the end points of sampled edge to induce a graph
  unordered_map<int, int> random_nodes_set;
  int sample_n = 0;
  for (auto it = edge_vec.begin(); it < edge_vec.end(); it++) {
    if (random_nodes_set.size() > 100000) {
      break;
    }
    if (random_nodes_set.find(it->first) == random_nodes_set.end()) {
      random_nodes_set[it->first] = sample_n;
      sample_n++;
    }
    if (random_nodes_set.find(it->second) == random_nodes_set.end()) {
      random_nodes_set[it->second] = sample_n;
      sample_n++;
    }
  }

  //construct submatrix for sampled subgraph
  int sample_m = 0;
  int *degree = new int[sample_n];
  for (int i = 0; i < sample_n; i++) {
    degree[i] = 0;
  }
  BitMatrix adjMatrix_sample;
  adjMatrix_sample.ConBitMatrix(sample_n);
  for (auto &x: random_nodes_set) {
    for (int j = 0; j < g.degree[x.first]; j++) {
      unordered_map<int, int>::const_iterator got = random_nodes_set.find(g.AdjList[x.first][j]);
      if (got != random_nodes_set.end()) {
        adjMatrix_sample.Update(x.second, got->second);
        sample_m++;
        degree[x.second]++;
      }
    }
  }

  sample_m = sample_m / 2;
  int display_step = sample_m / 10; //steps for displaying precision
  cout << "sample graph n= " << sample_n << " m= " << sample_m << endl;


  // Network reconstruction on sampled graph using Embedding
  cout << "network reconstruction on sampled nodes using " << methodname << " Embedding" << endl;
  MatrixXd Usample(sample_n, d);
  MatrixXd Vsample(sample_n, d);
  for (auto &x: random_nodes_set) {
    Usample.row(x.second) = embU.row(x.first);
    Vsample.row(x.second) = embV.row(x.first);
  }
  clock_t start_sample_reconstruction = clock();

  MatrixXd appr_matrix_sample = Usample * Vsample.transpose();
  clock_t end_sample_reconstruction = clock();
  vector<pair<double, pair<int, int>>> all_appr_sample;
  for (int i = 0; i < sample_n - 1; i++) {
    for (int j = i + 1; j < sample_n; j++) {
      all_appr_sample.push_back(make_pair(appr_matrix_sample(i, j), make_pair(i, j)));
    }
  }
  nth_element(all_appr_sample.begin(), all_appr_sample.begin() + sample_m - 1, all_appr_sample.end(), maxScoreCmp);
  sort(all_appr_sample.begin(), all_appr_sample.begin() + sample_m - 1, maxScoreCmp);
  int predict_positive_number_appr_sample = 0;
  for (int k = 0; k < 10; k++) {
    for (int i = 0; i < display_step; i++) {
      if (adjMatrix_sample.Find(all_appr_sample[k * display_step + i].second.first, all_appr_sample[k * display_step + i].second.second)) {
        predict_positive_number_appr_sample += 1;
      }
    }
    cout << "Number of edges on sampled graph: "<< k *display_step+display_step << ", " << methodname << " Precision: " << predict_positive_number_appr_sample/ (double) (k *display_step+display_step) << endl;
  }
}

