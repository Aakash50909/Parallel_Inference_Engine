#include "softmax.h"
#include <algorithm>
#include <cmath>
#include <numeric>

void softmax(std::vector<float>& x,int rows, int cols)
{
	for(int i=0;i<rows;i++)
	{
		float* row=x.data() + i*cols;
		float max_val=*std::max_element(row,row+cols);
		float sum=0.0f;
		for(int j=0;j<cols;j++)
		{
			row[j]=expf(row[j] - max_val);
			sum+=row[j];
		}
		for(int j=0;j<cols;j++)
		row[j] /=sum;
	}
}
