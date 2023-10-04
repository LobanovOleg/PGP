#include <stdio.h>

__global__ void kernel(double *array1, double *array2, double *result, int size_array) {
    int absolute_index = blockIdx.x * blockDim.x + threadIdx.x;
    int offset = gridDim.x * blockDim.x;
    while (absolute_index < size_array) {
        result[absolute_index] = array1[absolute_index] < array2[absolute_index] ? array1[absolute_index] : array2[absolute_index];
        absolute_index += offset;
    }
}

int main() {
    
    long int size = 0;
    scanf("%ld", &size);

    double *array1 = (double *)malloc(sizeof(double) * size);
    double *array2 = (double *)malloc(sizeof(double) * size);
    double *result = (double *)malloc(sizeof(double) * size);

    for (int i = 0; i < size; ++i) {
        scanf("%lf", &array1[i]);
    }

    for (int i = 0; i < size; ++i) {
        scanf("%lf", &array2[i]);
    }

    double *dev_arr1, *dev_arr2, *dev_res;
    cudaMalloc(&dev_arr1, sizeof(double) * size);
    cudaMalloc(&dev_arr2, sizeof(double) * size);
    cudaMalloc(&dev_res, sizeof(double) * size);
    cudaMemcpy(dev_arr1, array1, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_arr2, array2, sizeof(double) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_res, result, sizeof(double) * size, cudaMemcpyHostToDevice);

    kernel<<< 1024, 1024 >>>(dev_arr1, dev_arr2, dev_res, size);

    cudaMemcpy(result, dev_res, sizeof(double) * size, cudaMemcpyDeviceToHost);
    for (int i; i < size; i++) {
        printf("%.10lf ", result[i]);
    }
    printf("\n");

    cudaFree(dev_arr1);
    cudaFree(dev_arr2);
    cudaFree(dev_res);
    free(array1);
    free(array2);
    free(result);

    return 0;
}
