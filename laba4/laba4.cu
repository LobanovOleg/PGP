#include <iostream>
#include <math.h>
#include <cmath>
#include <stdbool.h>
#include <stdio.h>
#include <stdlib.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>

#define CSC(call)  													\
do {																\
	  cudaError_t res = call;											\
	  if (res != cudaSuccess) {										\
		  fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				  __FILE__, __LINE__, cudaGetErrorString(res));		\
		  exit(0);													\
	  }																\
} while(0)

class Comparator{
public:
    __host__ __device__ bool operator()(const double a, const double b) const{
        return fabs(a) < fabs(b);
    }
};

__global__ void kernel(double * A, double* E, int n, int i, int maximum) {
    int xdx = blockIdx.x * blockDim.x + threadIdx.x;
    int offsetx = gridDim.x * blockDim.x;
    for (int l = xdx; l < n; l += offsetx) {
        thrust::swap(A[l * n + i], A[l * n + maximum]);
        thrust::swap(E[l * n + i], E[l * n + maximum]);
    }
}

__global__ void Triangle_1(double *A, double *E, int n, int number) {
    int xdx = blockIdx.x * blockDim.x + threadIdx.x;
    int ydy = blockIdx.y * blockDim.y + threadIdx.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    double check;
    for (int i = number + 1 + xdx; i < n; i += offsetx) {
        check = -A[number * n + i]/A[number * n + number];
        for (int j = number + 1 + ydy; j < n; j += offsety) {
            A[j * n + i] = check * A[j * n + number] +A[j * n + i];
        }
        for (int j = ydy; j < n; j += offsety) {
            E[j * n + i] = check * E[j * n + number] + E[j * n + i];
        }
    }
}

__global__ void Triangle_2(double *A, double *E, int n, int number) {
    int xdx = threadIdx.x + blockIdx.x * blockDim.x;
    int ydy = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    double check;
    for (int i = number - 1 - xdx; i >= 0; i -= offsetx) {
        check = -A[number * n + i] /A[number * n + number];
        for (int j = ydy; j < n; j += offsety) {
            E[j * n + i] = check * E[j * n + number] + E[j * n + i];
        }
    }
}

__global__ void BasedKernel(double *A, double *E, int n) {
    int xdx = threadIdx.x + blockIdx.x * blockDim.x;
    int ydy = threadIdx.y + blockIdx.y * blockDim.y;
    int offsetx = gridDim.x * blockDim.x;
    int offsety = gridDim.y * blockDim.y;
    for (int i = xdx; i < n; i += offsetx) {
        for (int j = ydy; j < n; j += offsety) {
            E[j * n + i] = E[j * n + i] / A[i * n + i];
        }
    }
}

int main() {

    std::ios_base::sync_with_stdio(false);
    std::cin.tie(NULL);

    int n;
    scanf("%d", &n);
    if( n <= 0 ){
        return 0;
    }
    double *A = (double*)malloc( n * n * sizeof(double));
    double *E = (double*)malloc( n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            scanf("%lf", &A[n * j + i]);
        }
    }
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
          if (i == j) {
              E[i * n + j] = 1.0;
          } else {
              E[i * n + j] = 0.0;
          }
        }
    }
    double *dev_matrix_1;
    double *dev_matrix_2;
    int size = n * n * sizeof(double);
    CSC(cudaMalloc(&dev_matrix_1, size));
    CSC(cudaMalloc(&dev_matrix_2, size));
    CSC(cudaMemcpy(dev_matrix_1, A, size, cudaMemcpyHostToDevice));
    CSC(cudaMemcpy(dev_matrix_2, E, size, cudaMemcpyHostToDevice));
    int maximum;
    const Comparator comp;
    thrust::device_ptr<double> ptr_1 = thrust::device_pointer_cast(dev_matrix_1);

    for (int i = 0; i < n - 1; i++) {
        maximum = thrust::max_element(ptr_1 +i + i*n, ptr_1 + n * (i + 1), comp) - ptr_1 - i*n;

        if(maximum != i) {
            kernel<<< 256, 256 >>>(dev_matrix_1, dev_matrix_2, n , i, maximum);
            CSC(cudaGetLastError());
        }
        Triangle_1<<< dim3(16, 16), dim3(16, 16) >>>(dev_matrix_1, dev_matrix_2, n, i);
        CSC(cudaGetLastError());
    }
    for (int number = n -1; number > 0; number--) {
        Triangle_2<<< dim3(16, 16), dim3(16, 16) >>>(dev_matrix_1, dev_matrix_2, n, number);
        CSC(cudaGetLastError());
    }
    BasedKernel<<< dim3(16, 16), dim3(16, 16) >>>(dev_matrix_1, dev_matrix_2, n);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(A, dev_matrix_1, size, cudaMemcpyDeviceToHost));
    CSC(cudaMemcpy(E, dev_matrix_2, size, cudaMemcpyDeviceToHost));
    CSC(cudaFree(dev_matrix_1));
    CSC(cudaFree(dev_matrix_2));

    for(int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            printf("%.10lf ", E[j*n + i]);
        }
        printf("\n");
    }
    free(A);
    free(E);
    return 0;
}