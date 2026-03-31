#include <iostream>
#include <vector>
#include <cmath>
#include "matmul.h"

int main()
{
	std::vector<float> A = {1, 2, 3,
                             4, 5, 6};
	std::vector<float> B = {7,  8,
                             9,  10,
                             11, 12};
	std::vector<float> C;
	matmul(A, B, C, 2, 3, 2);
	std::vector<float> expected = {58, 64, 139, 154};
	bool passed = true;
   	for (int i = 0; i < 4; i++) {
        	if (fabsf(C[i] - expected[i]) > 1e-4f) {
            	passed = false;
        }
    }
	if (passed)
        std::cout << "matmul test PASSED\n";
    	else
        std::cout << "matmul test FAILED\n";
	// print the result matrix
    	std::cout << "C = \n";
    	std::cout << C[0] << " " << C[1] << "\n";
    	std::cout << C[2] << " " << C[3] << "\n";
	return 0;
}
