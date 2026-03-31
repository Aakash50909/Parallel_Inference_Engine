#include <iostream>
#include <vector>
#include <cmath>
#include "softmax.h"

int main() 
{
   	std::vector<float> x = {1.0f, 2.0f, 3.0f};
	softmax(x,1,3);
	std::cout << "Softmax([1,2,3]) = ";
    	for (float v : x)
        	std::cout << v << " ";
    	std::cout << "\n";
	float sum=0.0f;
	for(float v:x) sum+=v;
	std::cout<<"Sum should be 1.0 :"<<sum<<"\n";
	return 0;
}
