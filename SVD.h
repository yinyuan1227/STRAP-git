#include "Eigen/Sparse"
#include "Eigen/Dense"
#include <cstdlib>
#include <cmath>
#include <random>

namespace SVD
{	
	template<typename MatrixType>
	inline void QR_Decomposition(MatrixType& Q)
	{
		static const double EPS(1E-4);		
		for(int i = 0; i < Q.cols(); ++i)
		{
			for(int j = 0; j < i; ++j)
			{
				double r = Q.col(i).dot(Q.col(j));
				Q.col(i) -= r * Q.col(j);
			}
			
			double norm = Q.col(i).norm();
			
			if(norm < EPS)
			{
				for(int k = i; k < Q.cols(); ++k)
					Q.col(k) -= Q.col(k);
				return;
			}
			Q.col(i) /= norm;
		}
	}
	
	template<typename _MatrixType>
	class SVD
	{
	public:
		typedef _MatrixType MatrixType;
		
		SVD() {}
		
		
		SVD(const MatrixType& A, const int r, const int iter)
		{
			

		  clock_t tQ_start = clock();
		  Eigen::MatrixXf Q = Eigen::MatrixXf::Random(A.rows(), r);			
		  clock_t tQ_end = clock();
		  //cout << "Generate Q time: "<< (tQ_end - tQ_start) / (double) CLOCKS_PER_SEC << endl;
		  clock_t t0 = clock();
		  Q = A* Q;
		  clock_t t1 = clock();
		  // cout << "Matrix Mult time: "<< (t1 - t0) / (double) CLOCKS_PER_SEC << endl;
		  QR_Decomposition(Q);
		  clock_t t2 = clock();
		  //cout   << "QR time: "<< (t2 - t1) / (double) CLOCKS_PER_SEC  << endl;
		  //int iter = 2;
		  for (int i = 0; i< iter; i++){
		    Q = A.transpose() *Q;
		    QR_Decomposition(Q);
		    Q = A*Q;
		    QR_Decomposition(Q);
		  }
		  Eigen::MatrixXf C = Q.transpose()*A;			
		  clock_t t3 = clock();
		  Eigen::JacobiSVD<Eigen::MatrixXf> svdOfC(C, Eigen::ComputeThinU | Eigen::ComputeThinV);
		  clock_t t4 = clock();
		  //cout << "N*d svd time: "<< (t4 - t3) / (double) CLOCKS_PER_SEC << endl;
		  m_matrixU = Q * svdOfC.matrixU();
		  m_vectorS = svdOfC.singularValues();
		  m_matrixV = svdOfC.matrixV();
		}
		Eigen::MatrixXf matrixU() const
		{
			return m_matrixU;
		}
		
		Eigen::VectorXf singularValues() const
		{
			return m_vectorS;
		}
		
		Eigen::MatrixXf matrixV() const
		{
			return m_matrixV;
		}
		
	private:
		Eigen::MatrixXf m_matrixU;
		Eigen::VectorXf m_vectorS;
		Eigen::MatrixXf m_matrixV;
	};
	

}

