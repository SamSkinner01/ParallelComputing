
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;


__global__ void multiplyMatrixByVector(int* mat, int* vec, int m, int n, int size, int* output) {
	
	// Get the row and column that need to multiply
	int row = threadIdx.y + blockIdx.y * blockDim.y;
	int col = threadIdx.x + blockIdx.x * blockDim.x;

	// While the row is less than all the rows (if need to get new batch than add all threads)
	while (row < m) {

		// As long as row is less than all rows and col is less than 1
		if (row < m && col < 1) {
			int sum = 0;
			// sum the product of row in matrix and column in vector
			for (int k = 0; k < n; k++)
				sum += mat[row * n + k] * vec[k + col];
			output[row + col] = sum;
		}
		row += blockDim.y * gridDim.y;
	}
}

int main(){
	// Declare rows, cols, and size
	int m, n, size;

	// Declare mat and vec
	int* mat, * vec;

	// Input
	cout << "Please enter the dimensions of the matrix (m x n): \n";
	cin >> m >> n;

	// Init matrix size
	size = m * n;
	
	cout << "Generating the matrix(m x n) and vector(n)...\n\n";

	// init matrix and vector for host and device
	mat = new int[size];
	vec = new int[n];
	int* dev_mat, * dev_vec;

	// randomize matrix and vector
	srand(time(NULL));
	for (int i = 0; i < size; i++) {
		mat[i] = rand() % 10;
		if (i < n)
			vec[i] = rand() % 10;
	}

	// init ouput for host and device
	int* output = new int[m];
	int* dev_output;
	
	// Init device matrix, vec, and output vec
	cudaMalloc(&dev_mat, size * sizeof(int));
	cudaMalloc(&dev_vec, n * sizeof(int));
	cudaMalloc(&dev_output, m * sizeof(int));

	// Copy content from host to device
	cudaMemcpy(dev_mat, mat, size * sizeof(int), cudaMemcpyHostToDevice);
	cudaMemcpy(dev_vec, vec, n * sizeof(int), cudaMemcpyHostToDevice);

	// Define dimensions of grid and block
	dim3 blocks(32, 32, 1);
	dim3 grid(20, 20, 1);

	// Call function
	multiplyMatrixByVector << < grid, blocks >> > (dev_mat, dev_vec, m, n, size, dev_output);

	// Copy output
	cudaMemcpy(output, dev_output, m * sizeof(int), cudaMemcpyDeviceToHost);

	// If size <= 100 print out matrix, vector, and output
	if (m * n <= 100) {
		printf("Printing Matrix 1: \n");
		for (int i = 0; i < m; i++) {
			for (int j = 0; j < n; j++)
				printf("%d ", mat[i * n + j]);
			printf("\n");
		}

		printf("\nPrinting Vector: \n");
		for (int i = 0; i < n; i++)
			printf("%d\n", vec[i]);

		printf("\n\nOutput Vector: \n");
		for (int i = 0; i < m; i++)
			printf("%d ", output[i]);
		printf("\n");
	}

	printf("\nFINISHED CALCULATING\n");
}

