#include <vector>
#include <stdexcept>
#include <cmath>
#include "matmul.h"

void matmul(const std::vector<float>& A, const std::vector<float>& B,std::vector<float>& C, int M, int K, int N)
{
	if((int)A.size() !=M*K)
		throw std::invalid_argument("matmul: A size mismatch");
	if((int)B.size()!=K*N)
		throw std::invalid_argument("matmul: B size mismatch");
	C.assign(M*N,0.0f);
	for(int i=0;i<M;i++)
	{
		for(int j=0;j<N;j++)
		{
			float sum= 0.0f;
			for(int k=0;k<K;k++)
			{
				sum+=A[i*K+k]*B[k*N+j];
			}
			C[i*N+j]=sum;
		}
	}
}

