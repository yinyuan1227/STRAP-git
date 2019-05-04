#include <algorithm>
#include <iostream>
#include <string>
#include <sstream>
#include <cstdlib>
#include <vector>
#include <unordered_map>
#include "Graph.h"
#include <fstream>
#include <cstring>


bool maxScoreCmp(const pair<double, pair<int, int>>& a, const pair<double, pair<int, int>>& b){
    return a.first > b.first;
}

bool maxScoreCmpPair(const pair<int, double>& a, const pair<int, double>&  b){
    return a.second < b.second;
}





int main(int argc,  char **argv){
  srand((unsigned)time(0));
  char *endptr;
  string queryname = argv[1];
  double percent = strtod(argv[2], &endptr);

  string dataset =  "NR_Dataset/" + queryname + ".txt";
  string traindataset =  "LP_Dataset/Train/" + queryname + ".txt";
  string ptestdataset =  "LP_Dataset/Positive/" + queryname + ".txt";
  string ntestdataset = "LP_Dataset/Negative/" + queryname + ".txt";
  UGraph g;
  clock_t t0 = clock();
  g.inputGraph(dataset);
  clock_t t1 = clock();
  cout<<"reading in graph takes "<<(t1-t0)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;
  clock_t t2 = clock();
  g.RandomSplitGraph(dataset, ptestdataset, traindataset, percent);
  clock_t t3 = clock();
  cout<<"splitting graph takes "<<(t3-t2)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;
  g.NegativeSamples(ntestdataset, percent);
  clock_t t4 = clock();
  cout<<"sampling negative edges takes "<<(t4-t3)/(1.0*CLOCKS_PER_SEC)<<" s."<<endl;

}

