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
  std::vector<float> values;
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
  Graph g;
  g.inputGraph(dataset);
  MatrixXf U = load_csv<MatrixXf>(inUfile);
  MatrixXf V = load_csv<MatrixXf>(inVfile);
  int d = U.cols();

  // if graph size > 100000, use sampled graph, otherwise use original graph
  // sample a subgraph using random edge induction method
  srand((unsigned) time(0));
  cout << "Constructing sample graph using edge induction..." << endl;
  int sample_init_m = g.m;
  if (g.n >= 100000) {
    sample_init_m = min(100000, g.m);
  }

  // uniform sample sample_init_m edges
  int *edge_array = new int[2 * g.m];
  int edge_index = 0;
  for (int i = 0; i < g.n; i++) {
    for (int j = 0; j < g.outdegree[i]; j++) {
      edge_array[edge_index] = i;
      edge_array[edge_index + 1] = g.outAdjList[i][j];
      edge_index += 2;
    }
  }

  for (int i = 0; i < sample_init_m; i++) {
    int r = rand() % (g.m - i);
    int temp1 = edge_array[2 * i + 2 * r];
    int temp2 = edge_array[2 * i + 2 * r + 1];
    edge_array[2 * i + 2 * r] = edge_array[2 * i];
    edge_array[2 * i + 2 * r + 1] = edge_array[2 * i + 1];
    edge_array[2 * i] = temp1;
    edge_array[2 * i + 1] = temp2;
  }

  // use the end points of sampled edge to induce a graph
  unordered_map<int, int> random_nodes_set;
  int sample_n = 0;
  for (int i = 0; i < 2 * sample_init_m; i++) {
    if (random_nodes_set.size() > 100000) {
      break;
    }
    if (random_nodes_set.find(edge_array[i]) == random_nodes_set.end()) {
      random_nodes_set[edge_array[i]] = sample_n;
      sample_n++;
    }
  }

  // construct submatrix for sampled subgraph
  int sample_m = 0;
  BitMatrix adjMatrix_sample;
  adjMatrix_sample.ConBitMatrix(sample_n);
  for (auto &x: random_nodes_set) {
    for (int j = 0; j < g.outdegree[x.first]; j++) {
      unordered_map<int, int>::const_iterator got = random_nodes_set.find(g.outAdjList[x.first][j]);
      if (got != random_nodes_set.end()) {
        adjMatrix_sample.Update(x.second, got->second);
        sample_m++;
      }
    }
  }

  int display_step = sample_m / 10; //steps for displaying precision
  cout << "sample graph n= " << sample_n << " m= " << sample_m << endl;


  // Network reconstruction on sampled graph using Embedding
  cout << "network reconstruction on sampled nodes using " << methodname << " Embedding" << endl;
  MatrixXf Usample(sample_n, d);
  MatrixXf Vsample(sample_n, d);
  for (auto &x: random_nodes_set) {
    Usample.row(x.second) = U.row(x.first);
    Vsample.row(x.second) = V.row(x.first);
  }
  clock_t start_sample_reconstruction = clock();

  MatrixXf appr_matrix_sample = Usample * Vsample.transpose();
  clock_t end_sample_reconstruction = clock();
  vector<pair<double, pair<int, int>>> all_appr_sample;
  for (int i = 0; i < sample_n; i++) {
    for (int j = 0; j < sample_n; j++) {
      if (i != j) {
        all_appr_sample.push_back(make_pair(appr_matrix_sample(i, j), make_pair(i, j)));
      }
    }
  }
  nth_element(all_appr_sample.begin(), all_appr_sample.begin()+sample_m-1, all_appr_sample.end(), maxScoreCmp);
  sort(all_appr_sample.begin(), all_appr_sample.begin()+sample_m-1, maxScoreCmp);
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

