#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <stdio.h>
#include <iostream>
using namespace std;

#define threadsPerBlock 1024	// Threads Per Block
#define imin(a,b) (a<b?a:b)		

__global__ void findMean(int* a, int N, double* mean) {
	// Device shared memory to find the sum
	__shared__ int sum[threadsPerBlock];

	// Calc local tid
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	
	// Has own shared memory index based on its current index in the block
	int sumIndex = threadIdx.x;

	// Stores current element in batch
	double temp = 0;
	while (tid < N) {
		temp += a[tid];
		tid += blockDim.x * gridDim.x;
	}

	// Shared memory at the index contains the sum of the batch
	sum[sumIndex] = temp;

	// Sync threads before reduction
	__syncthreads();

	// Reduce
	int i = blockDim.x / 2;
	while (i != 0) {
		if (sumIndex < i)
			sum[sumIndex] += sum[sumIndex + i];
		__syncthreads();
		i /= 2;
	}

	// Find mean over blocks sum
	if (sumIndex == 0) {
		mean[blockIdx.x] = sum[0];
		mean[blockIdx.x] /= N;
	}
}

__global__ void distanceFromMeanSquared(int* a, int N, double mean, double* distance) {
	// Shared memory to find the squared distance from the mean
	__shared__ double distanceFromMean[threadsPerBlock];
	
	// Calc local tid
	int tid = threadIdx.x + blockIdx.x * blockDim.x;

	// Has own shared memory index based on its current index in the block 
	int distanceIndex = threadIdx.x;
	
	// Holds squared distance from mean for each batch
	double temp = 0;
	while (tid < N) {
		temp += pow((a[tid]-mean),2);
		tid += blockDim.x * gridDim.x;
	}

	// Fill shared memory with squared distance from mean
	distanceFromMean[distanceIndex] = temp;

	// Sync threads before reduction
	__syncthreads();

	// Reduction
	int i = blockDim.x / 2;
	while (i != 0) {
		if (distanceIndex < i)
			distanceFromMean[distanceIndex] += distanceFromMean[distanceIndex + i];
		__syncthreads();
		i /= 2;
	}

	// Find distance from mean squared for the block
	if (distanceIndex == 0) {
		distance[blockIdx.x] = distanceFromMean[0];
	}
}

int main(){
	// Size
	int N;
	
	// Declare vector
	int* a;
	
	// Init vars for mean and variance
	double var = 0, mean = 0;
	
	// Declare device vector
	int* dev_a;

	// User input
	cout << "Enter the size of the dataset: \n";
	cin >> N;

	// Initialize vector of size N
	a = new int[N];

	// Randomly generate data inside of the vector
	srand(time(NULL));
	printf("\n\nRandomized Dataset: \n\n");
	for (int i = 0; i < N; i++) {
		a[i] = rand() % 10;
		// Print nicely :)
		if(i == N-1)
			printf("%d", a[i]);
		else
			printf("%d, ", a[i]);
	}

	// Initialize the device vector
	cudaMalloc(&dev_a, N * sizeof(int));
	
	// Copy randomly generated vector into device vector
	cudaMemcpy(dev_a, a, N * sizeof(int), cudaMemcpyHostToDevice);

	// Generate the number of blocks per grid given threads per block 
	const int blocksPerGrid =
		imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

	// Create partial mean and dev_partial mean of size blocksPerGrid
	double* partial_mean = new double[blocksPerGrid];
	double* dev_partial_mean;

	// Init dev_partial_mean on device
	cudaMalloc(&dev_partial_mean, blocksPerGrid * sizeof(double));

	// Find partial mean
	findMean << < blocksPerGrid, threadsPerBlock >> > (dev_a, N, dev_partial_mean);
	
	// Copy partial mean
	cudaMemcpy(partial_mean, dev_partial_mean, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);

	// Iterate through the partial means to calculate the global mean
	for (int i = 0; i < blocksPerGrid; i++) {
		mean += partial_mean[i];
	}
	mean /= blocksPerGrid;

	// Declare and Initialize partial squared distances from the mean
	double distance = 0;
	double* partial_distance = new double[blocksPerGrid];
	double* dev_partial_distance;
	cudaMalloc(&dev_partial_distance, blocksPerGrid*sizeof(double));

	// Call the function
	distanceFromMeanSquared << < blocksPerGrid, threadsPerBlock >> > (dev_a, N, mean, dev_partial_distance);

	// Copy partial squared distances from the mean
	cudaMemcpy(partial_distance, dev_partial_distance, blocksPerGrid*sizeof(double), cudaMemcpyDeviceToHost);
	
	// Calclulate global distances
	for (int i = 0; i < blocksPerGrid; i++) 
		distance += partial_distance[i];

	// Final variance calculation.
	var = distance / N;

	cout << "\n\nThe mean of this dataset is: " << mean << endl;
	cout << "The variance of this dataset is: " << var << endl;
}

