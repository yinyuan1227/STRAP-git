#include <algorithm>
#include <iostream>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <deque>
#include <unordered_map>
#include "Graph.h"
#include "SVD.h"
#include <fstream>
#include <cstring>
#include <thread>
#include <mutex>
#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <chrono>
#include <climits>

using namespace Eigen;


bool maxScoreCmpTriplet(const Triplet<double>& a, const Triplet<double>& b){
  return a.value() > b.value();
}

void BackwardSearch(int* random_w, int start, int end, Graph* g, double alpha, double residuemax, double reservemin, vector<Triplet<double>>* answer){
  // Computing all node PPR on G
  double* Residue = new double[g->n];
  double* Reserve = new double[g->n];
  Node_Set* candidate_set = new Node_Set(g->n);
  Node_Set* touch_set = new Node_Set(g->n);

  for (int i = 0; i < g->n; i++){
    Residue[i]=0;
    Reserve[i]=0;
  }
  for(int it = start; it < end; it++){
    int w = random_w[it];
    Residue[w] = 1;
    candidate_set->Push(w);
    touch_set->Push(w);
    while(candidate_set->KeyNumber !=0){
      int tempNode = candidate_set->Pop();
      touch_set->Push(tempNode);
      int inSize = g->indegree[tempNode];
      double tempResidue = Residue[tempNode];
      Residue[tempNode] = 0;
      Reserve[tempNode] += alpha* tempResidue;

      for(int k = 0; k < inSize; k++){
        int newNode = g->inAdjList[tempNode][k];
        Residue[newNode] += tempResidue * (1-alpha) / (double)g->outdegree[newNode];
        touch_set->Push(newNode);
        if(Residue[newNode] > residuemax){
          candidate_set->Push(newNode);
        }
      }
    }
    
    for (int k = 0; k < touch_set->KeyNumber; k++){
      if (w != touch_set->HashKey[k]){
        if (Reserve[touch_set->HashKey[k]] > reservemin){
          answer->push_back(Triplet<double>(touch_set->HashKey[k],w, Reserve[touch_set->HashKey[k]]));
        }
      }
      Reserve[touch_set->HashKey[k]] = 0;
      Residue[touch_set->HashKey[k]] = 0;
    }
    candidate_set->Clean();
    touch_set->Clean();
  }
}


void BackwardSearchT(int* random_w, int start, int end, Graph* g, double alpha, double residuemax, double reservemin, vector<Triplet<double>>* answer){
  double* Residue = new double[g->n];
  double* Reserve = new double[g->n];
  Node_Set* candidate_set = new Node_Set(g->n);
  Node_Set* touch_set = new Node_Set(g->n);


  for (int i = 0; i < g->n; i++){
    Residue[i]=0;
    Reserve[i]=0;
  }
  for(int it = start; it < end; it++){
    int w = random_w[it];
    Residue[w] = 1;
    candidate_set->Push(w);
    touch_set->Push(w);
    while(candidate_set->KeyNumber !=0){
      int tempNode = candidate_set->Pop();
      touch_set->Push(tempNode);
      int outSize = g->outdegree[tempNode];
      double tempResidue = Residue[tempNode];
      Residue[tempNode] = 0;
      Reserve[tempNode] += alpha* tempResidue;

      for(int k = 0; k < outSize; k++){
        int newNode = g->outAdjList[tempNode][k];
        Residue[newNode] += tempResidue * (1-alpha) / (double)g->indegree[newNode];
        touch_set->Push(newNode);
        if(Residue[newNode] > residuemax){
          candidate_set->Push(newNode);
        }
      }
    }

    for (int k = 0; k < touch_set->KeyNumber; k++){
      if (w != touch_set->HashKey[k]){
        if(Reserve[touch_set->HashKey[k]] > reservemin){
          answer->push_back(Triplet<double>(w,touch_set->HashKey[k],Reserve[touch_set->HashKey[k]]));
        }
      }
      Reserve[touch_set->HashKey[k]] = 0;
      Residue[touch_set->HashKey[k]] = 0;
    }
    candidate_set->Clean();
    touch_set->Clean();
  }
}



int main(int argc,  char **argv){
  auto start_time = std::chrono::system_clock::now();
  srand((unsigned)time(0));
  char *endptr;
  string queryname = argv[1];
  string querypath = argv[2];
  string EBpath = argv[3];
  string dataset = querypath  + queryname + ".txt";
  string outUfile = EBpath + queryname + "_strap_svd_d_U.csv";
  string outVfile = EBpath + queryname + "_strap_svd_d_V.csv";
  ofstream outU(outUfile.c_str());
  ofstream outV(outVfile.c_str());
  Graph g;
  g.inputGraph(dataset);
  clock_t start = clock();
  double alpha = strtod(argv[4], &endptr);
  int pass = strtod(argv[5], &endptr);
  double backward_theta = strtod(argv[6], &endptr);
  int NUMTHREAD = strtod(argv[7], &endptr);;
  double residuemax = backward_theta; // PPR error up to residuemax
  double reservemin = backward_theta; // Retain approximate PPR larger than reservemin
  cout << "alpha: " << alpha << " residuemax: " << residuemax << " reservemin: " << reservemin <<endl;

  int d = 128;

  int* random_w = new int[g.n];
  for(int i = 0; i < g.n; i++){
    random_w[i] = i;
  }
  for(int i = 0; i < g.n; i++){
    int r = rand()%(g.n-i);
    int temp = random_w[i + r];
    random_w[i + r] = random_w[i];
    random_w[i] = temp;
  }


  cout << "ppr computation " << endl;
  auto ppr_start_time = std::chrono::system_clock::now();
  vector<thread> threads;
  vector<vector<Triplet<double>>> tripletList(NUMTHREAD);
  deque<Triplet<double>> TotalTripletList;

  for (int t = 1; t <= NUMTHREAD; t++){
    int start = (t-1)*(g.n/NUMTHREAD);
    int end = 0;
    if (t == NUMTHREAD){
      end = g.n;
    }
    else{
      end = t*(g.n/NUMTHREAD);
    }
    threads.push_back(thread(BackwardSearch, random_w, start, end, &g, alpha, residuemax, reservemin, &tripletList[t-1]));
  }
  for (int t = 0; t < NUMTHREAD ; t++){
    threads[t].join();
  }
  vector<thread>().swap(threads);

  for (int t = 1; t <= NUMTHREAD; t++){
    int start = (t-1)*(g.n/NUMTHREAD);
    int end = 0;
    if (t == NUMTHREAD){
      end = g.n;
    }
    else{
      end = t*(g.n/NUMTHREAD);
    }
    threads.push_back(thread(BackwardSearchT, random_w, start, end, &g, alpha, residuemax, reservemin, &tripletList[t-1]));
  }
  for (int t = 0; t < NUMTHREAD ; t++){
    threads[t].join();
  }
  vector<thread>().swap(threads);
  delete[] random_w;

  auto start_ppr_matrix_time = chrono::system_clock::now();
  auto elapsed_ppr_time = chrono::duration_cast<std::chrono::seconds>(start_ppr_matrix_time - ppr_start_time);
  cout << "computing ppr time: "<< elapsed_ppr_time.count() << endl;


  long total_size = 0;
  for (int t = 0; t < NUMTHREAD; t++){
    total_size += tripletList[t].size();
  }
  cout << "total size: " << total_size << endl;
  cout << "compute  TotalTripletList" <<endl;
  for (int t = 0; t < NUMTHREAD; t++){
    TotalTripletList.insert(TotalTripletList.end(), tripletList[t].begin(), tripletList[t].end());
    vector<Triplet<double>>().swap(tripletList[t]);
  }
  vector<vector<Triplet<double>>>().swap(tripletList);

  long nnz = TotalTripletList.size();
  cout << "nnz1 + nnz2 = " << nnz << endl;
  auto merge_ppr_time = chrono::system_clock::now();
  auto elapsed_merge_ppr_time = chrono::duration_cast<std::chrono::seconds>(merge_ppr_time - start_ppr_matrix_time);
  cout << "merge ppr vec time: "<< elapsed_merge_ppr_time.count() << endl;


  //Combine ppr and tppr vector into Eigen:Sparse
  long max_nnz = INT_MAX;
  if (TotalTripletList.size() > max_nnz){
    nth_element(TotalTripletList.begin(), TotalTripletList.begin()+max_nnz, TotalTripletList.end(), maxScoreCmpTriplet);
    TotalTripletList.erase(TotalTripletList.begin()+max_nnz+1, TotalTripletList.end());
    nnz = max_nnz;
  }

  cout << "deque to sparse matrix" << endl;
  // Rewrite SparseMatrix::setFromTriplets to optimize memory usage
  SparseMatrix<float> ppr_matrix(g.n, g.n);
  SparseMatrix<float> trMat(g.n, g.n);
  deque<Triplet<double>>::iterator it;
  long max_step = nnz / 2 + 1;
  long step = 0;
  while (nnz){
    SparseMatrix<float>::IndexVector wi(trMat.outerSize());
    wi.setZero();
    if (nnz < max_step){
      step = nnz;
      nnz = 0;
    } else{
      step = max_step;
      nnz -= max_step;
    }

    for (int j = 0; j < step; j++){
      wi(TotalTripletList[j].col())++;
    }

    trMat.reserve(wi);

    for (int j = 0; j < step; j++){
      it = TotalTripletList.begin();
      trMat.insertBackUncompressed(it->row(), it->col()) = (float)log10(it->value() / reservemin);
      TotalTripletList.erase(TotalTripletList.begin());
    }

    trMat.collapseDuplicates(internal::scalar_sum_op<float, float>());
  }
  deque<Triplet<double>>().swap(TotalTripletList);

  ppr_matrix = trMat;
  trMat.resize(0, 0);
  trMat.data().squeeze();


  auto svd_start_time = chrono::system_clock::now();
  auto elapsed_deque_to_sparse_time = chrono::duration_cast<std::chrono::seconds>(svd_start_time - merge_ppr_time);
  cout << "deque to sparse time: "<< elapsed_deque_to_sparse_time.count() << endl;
  cout << "start svd..." << endl;


  // Compute SVD
  SVD::SVD<SparseMatrix<float>> svd(ppr_matrix, d, pass);
  auto end_eb_time = chrono::system_clock::now();
  auto elapsed_svd_time = chrono::duration_cast<std::chrono::seconds>(end_eb_time - svd_start_time);
  cout << "svd time: "<< elapsed_svd_time.count() << endl;

  VectorXf vS = svd.singularValues();
  double ppr_norm = ppr_matrix.norm()*ppr_matrix.norm();
  double S_norm = vS.norm()*vS.norm();
  cout << S_norm << " " << ppr_norm << " ratio: " << (double)S_norm/ppr_norm << endl;

  MatrixXf S = vS.cwiseSqrt().asDiagonal();
  MatrixXf U = svd.matrixU()*S;
  MatrixXf V = svd.matrixV()*S;
  outU << U.format(CSVFormat);
  outV << V.format(CSVFormat);
  auto end_time = chrono::system_clock::now();
  auto elapsed_write_time = chrono::duration_cast<std::chrono::seconds>(end_time - end_eb_time);
  cout << "write out embedding time: "<< elapsed_write_time.count() << endl;
  auto elapsed_time = chrono::duration_cast<std::chrono::seconds>(end_time - start_time);
  cout << "total embedding time: "<< elapsed_time.count() << endl;

}

