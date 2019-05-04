icc -O3 -mkl -pthread -march=core2 -std=c++11 -no-multibyte-chars -qopenmp  frpca.c matrix_vector_functions_intel_mkl.c matrix_vector_functions_intel_mkl_ext.c strap_frpca_u.cpp -o STRAP_FRPCA_U
icc -O3 -mkl -pthread -march=core2 -std=c++11 -no-multibyte-chars -qopenmp  frpca.c matrix_vector_functions_intel_mkl.c matrix_vector_functions_intel_mkl_ext.c strap_frpca_d.cpp -o STRAP_FRPCA_D
g++ -pthread -march=core2 -std=c++11 -O3 -o STRAP_SVD_U strap_svd_u.cpp
g++ -pthread -march=core2 -std=c++11 -O3 -o STRAP_SVD_D strap_svd_d.cpp
g++ -march=core2 -std=c++11 -O3 -o NET_RE_U net_re_u.cpp
g++ -march=core2 -std=c++11 -O3 -o NET_RE_D net_re_d.cpp
g++ -march=core2 -std=c++11 -O3 -o GEN_DATA_U gen_data_u.cpp
g++ -march=core2 -std=c++11 -O3 -o GEN_DATA_D gen_data_d.cpp
g++ -march=core2 -std=c++11 -O3 -o LINK_PRE_U link_pre_u.cpp
g++ -march=core2 -std=c++11 -O3 -o LINK_PRE_D link_pre_d.cpp

