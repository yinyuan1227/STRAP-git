#include <algorithm>
#include <iostream>
#include <sstream>
#include <vector>
#include <unordered_set>
#include "Graph.h"
#include "Eigen/Dense"
#include "SVD.h"
#include <boost/functional/hash.hpp>

using namespace Eigen;

template<typename M>
M load_csv (const std::string & path) {
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

bool maxScoreCmp(const pair<double, pair<int, int>>& a, const pair<double, pair<int, int>>& b){
    return a.first > b.first;
}


int main(int argc,  char **argv){
  srand((unsigned)time(0));
  char *endptr;
  string queryname = argv[1];
  string methodname = argv[2];
  string ptestdataset =  "LP_Dataset/Positive/" + queryname + ".txt";
  string ntestdataset = "LP_Dataset/Negative/" + queryname + ".txt";
  string inUfile = "LP_EB/" + queryname + "_" + methodname + "_U.csv";
  string inVfile = "LP_EB/" + queryname + "_" + methodname + "_V.csv";
  ifstream ptest(ptestdataset.c_str());
  ifstream ntest(ntestdataset.c_str());
  unordered_set<pair<int, int>, boost::hash< pair<int, int>>> pedge_set;
  unordered_set<pair<int, int>, boost::hash< pair<int, int>>> nedge_set;

  // Load in positive and negative test sets
  int n = 0;
  ptest >> n;
  ntest >> n;
  int sample_m = 0;
  while(ptest.good()){
    int from;
    int to;
    ptest >> from >> to;
    pedge_set.insert(make_pair(from, to));
    sample_m++;
  }
  while(ntest.good()){
    int from;
    int to;
    ntest >> from >> to;
    nedge_set.insert(make_pair(from, to));
  }

  // Load in embeddings
  MatrixXf U = load_csv<MatrixXf>(inUfile);
  MatrixXf V = load_csv<MatrixXf>(inVfile);

  // Compute predicted score
  int d = U.cols();
  vector<pair<double, pair<int, int>>> embedding_score;
  for (auto it = pedge_set.begin(); it != pedge_set.end(); ++it) {
    int i = it->first;
    int j = it->second;
    double score = U.row(i).dot(V.row(j));    
    embedding_score.push_back(make_pair(score, make_pair(i,j)));
  }
  for (auto it = nedge_set.begin(); it != nedge_set.end(); ++it) {
    int i = it->first;
    int j = it->second;
    double score = U.row(i).dot(V.row(j));    
    embedding_score.push_back(make_pair(score, make_pair(i,j)));
  }

  // Top sample_m predicted edges is considered
  nth_element(embedding_score.begin(), embedding_score.begin()+sample_m-1, embedding_score.end(), maxScoreCmp);
  sort(embedding_score.begin(), embedding_score.begin()+sample_m-1, maxScoreCmp);
  int predict_positive_number = 0;
  for (auto it = embedding_score.begin(); it != embedding_score.begin()+sample_m; ++it) {
    int i = it->second.first;
    int j = it->second.second;
    if(pedge_set.find(make_pair(i,j)) != pedge_set.end()){
      predict_positive_number ++;
    }
  }
  cout << methodname << " link prediction precision: " << predict_positive_number/ (double) (sample_m) << endl;
}


