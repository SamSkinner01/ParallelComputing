#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <stdio.h>
using namespace std;

#define threadsPerBlock 1024
#define imin(a,b) (a<b?a:b)

__global__ void findLargest(int* vec, int N, int* largest) {
	// Shared memory to store the largest values in blocks
	__shared__ double cache[threadsPerBlock];

	// Get tid and cache index
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int cacheIndex = threadIdx.x;

	// If the value at vec[tid] > temp, store it 
	// we want the largest numbers anyways
	int temp = 0;
	while (tid < N) {
		if(vec[tid] > temp)
			temp = vec[tid];
		tid += blockDim.x * gridDim.x;
	}

	// store that value in the shared memory and sync threads
	cache[cacheIndex] = temp;
	__syncthreads();

	// start reduction
	int i = blockDim.x / 2;
	while (i != 0) {
		if (cacheIndex < i)
			// reduce by swapping the indices in the shared memory if the side on 
			// the right is larger
			if (cache[cacheIndex] < cache[cacheIndex + i]) {
				int swap = cache[cacheIndex];
				cache[cacheIndex] = cache[cacheIndex + i];
				cache[cacheIndex + i] = swap;
			}

		__syncthreads();
		i /= 2;
	}

	// store the largest value found
	if (cacheIndex == 0)
		largest[blockIdx.x] = cache[0];
}

int main() {
	// declare size and vectors for host and device
	int N;
	int* vec;
	int* dev_vec;

	// input
	printf("Please enter the size N for the vector: ");
	cin >> N;
	
	// init vector
	vec = new int[N];
	
	// randomly generate the vector
	printf("\nRandomly Generating your vector...");
	srand(time(NULL));
	for (int i = 0; i < N; i++) 
		vec[i] = rand() % 1000;

	//print if N < 100
	if (N < 100) {
		printf("\n\nYour randomly generated vector is: \n");
		for (int i = 0; i < N; i++) {
			if (i < N - 1)
				printf("%d, ", vec[i]);
			else
				printf("%d", vec[i]);
		}
	}
	// else print first 20
	else {
		printf("\n\nThe first 20 elements of your randomly generated vector is: \n");
		for (int i = 0; i < 20; i++) {
			if (i < N - 1)
				printf("%d, ", vec[i]);
			else
				printf("%d", vec[i]);
		}
	}

	// init dev_vec on device
	cudaMalloc(&dev_vec, N * sizeof(int));

	// copy vec from host to device
	cudaMemcpy(dev_vec, vec, N * sizeof(int), cudaMemcpyHostToDevice);

	// calculate the number of blocks per grid
	const int blocksPerGrid =
		imin(32, (N + threadsPerBlock - 1) / threadsPerBlock);

	// delcare and init the partial largest on host and device
	int* partial_largest = new int[blocksPerGrid];
	int* dev_partial_largest;
	cudaMalloc(&dev_partial_largest, blocksPerGrid * sizeof(int));

	// call function
	findLargest << <blocksPerGrid, threadsPerBlock >> > (dev_vec, N, dev_partial_largest);

	// copy results
	cudaMemcpy(partial_largest, dev_partial_largest, blocksPerGrid * sizeof(int), cudaMemcpyDeviceToHost);

	// finalize finding largest on host
	int largest = partial_largest[0];
	for (int i = 0; i < blocksPerGrid; i++) {
		if (partial_largest[i] > largest)
			largest = partial_largest[i];
	}

	printf("\n\n\nThe largest number in the dataset was: %d\n", largest);
}


