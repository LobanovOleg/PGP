#include <stdio.h>
#include <math.h>
#include <iostream>

#define CSC(call)                   \
do {                                \
    cudaError_t res = call;         \
    if (res != cudaSuccess) {       \
        fprintf(stderr, "ERROR in %s:%d. Message: %s\n",            \
                __FILE__, __LINE__, cudaGetErrorString(res));       \
        exit(0);                    \
    }                               \
} while(0)

const int ogre = 16777216;

__global__ void Histohram(int* input, int* histogram, int size) {
	int xdx = blockDim.x * blockIdx.x + threadIdx.x;
	int offsetx = blockDim.x * gridDim.x;
	for (int i = xdx; i < size; i+= offsetx) {
		atomicAdd(&input[histogram[i]], 1);
	}
}

__global__ void ScanKernel(int* histogram, int* arr) {
	int kek = blockDim.x;
	__shared__ int SharedHistogram[1024];
    int check = 1;
	int xdx = blockDim.x * blockIdx.x + threadIdx.x;
	SharedHistogram[threadIdx.x] = histogram[xdx];
	__syncthreads();

	while (kek > check) {
		if (2 * check + threadIdx.x * 2 *check - 1 < kek) {
            SharedHistogram[2 * check + threadIdx.x * 2 * check - 1] += SharedHistogram[check+ threadIdx.x *  check * 2 - 1];
        }
        check= check* 2;
		__syncthreads();
	}
	int temp = 0;

	if (threadIdx.x == kek - 1) {
		temp = SharedHistogram[threadIdx.x];
		SharedHistogram[threadIdx.x] = 0;
	}
	check = check / 2;
	__syncthreads();
	
	while (check >= 1) {
		if (check* 2 * threadIdx.x + 2 * check - 1 < kek) {
			auto swap = SharedHistogram[2 *check * threadIdx.x + check - 1];
			SharedHistogram[check * 2 * threadIdx.x + check - 1] = SharedHistogram[check * 2 * threadIdx.x + 2 * check - 1];
			SharedHistogram[check * 2 * threadIdx.x + 2 * check - 1] += swap;
		}
		check = check /2;
		__syncthreads();
	}

	if (threadIdx.x == kek - 1) {
		histogram[xdx] = temp;
		arr[blockIdx.x] = temp;
	} else {
		histogram[xdx] = SharedHistogram[threadIdx.x + 1];
	}
}

__global__ void Shift(int* histogram, int* newArr) {
	int xdx = blockDim.x * blockIdx.x + threadIdx.x;
	if (0 < blockIdx.x)
		histogram[xdx] += newArr[blockIdx.x - 1];
}

void Scan(int* input, int size) {
	int maximum = fmax(1.0, ((double)size / 1024.0));
	int minimum = fmin((double)size, 1024.0);
	int* newInput;
	CSC(cudaMalloc((void**)&newInput, sizeof(int) * maximum));
	ScanKernel<<< maximum, minimum >>>(input, newInput);
	cudaDeviceSynchronize();
	if (size > 1024)
	{
		Scan(newInput, (size/ 1024));
		Shift<<<(size / 1024), 1024 >>>(input, newInput);
		cudaDeviceSynchronize();
	}
	cudaFree(newInput);
}

__global__ void CountSort(int* input, int* histogram, int* outPut, int size) {
	int xdx =  threadIdx.x + blockDim.x * blockIdx.x;
	int offsetx = gridDim.x * blockDim.x;
	for (int i = xdx; i < size; i +=offsetx)
	{
		outPut[atomicAdd(&input[histogram[i]], -1) - 1] = histogram[i];
	}
}

int main() {
	int size;
	fread(&size, sizeof(int), 1, stdin);
	auto input = new int[size];
	fread(input, sizeof(int), size, stdin);

	int* devInput;
    int* newInput;
    int* outPut;
	CSC(cudaMalloc((void**)&devInput, sizeof(int) * (ogre)));
	CSC(cudaMemset(devInput, 0, sizeof(int) * (ogre)));
	CSC(cudaMalloc((void**)&newInput, sizeof(int) * size));
	CSC(cudaMemcpy(newInput, input, sizeof(int) * size, cudaMemcpyHostToDevice));
	Histohram<<<256, 256>>>(devInput, newInput, size);
	cudaDeviceSynchronize();
	Scan(devInput, ogre);
	CSC(cudaMalloc((void**)&outPut, sizeof(int) * size));
	CountSort<<<256, 256>>>(devInput, newInput, outPut, size);
	cudaDeviceSynchronize();
	CSC(cudaMemcpy(input, outPut, sizeof(int) * size, cudaMemcpyDeviceToHost));
	fwrite(input, sizeof(int), size, stdout);

    return 0;
}