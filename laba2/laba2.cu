#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define CSC(call) \
    do { \
        cudaError_t res = call; \
        if (res != cudaSuccess) { \
            fprintf(stderr, "ERROR in %s:%d. Message: %s\n", \
            __FILE__, __LINE__, cudaGetErrorString(res)); \
            exit(0); \
            } \
    } while(0)

__device__ double grey(uchar4 check) {
    return (0.299 * double(check.x) + 0.587 * double(check.y) + 0.114 * double(check.z));
}

__device__ double Gradient(double Gx, double Gy) {
    int grad = rint(sqrt(Gx * Gx + Gy * Gy));
    return min(grad, 255);
}

__global__ void kernel(cudaTextureObject_t tex, uchar4 *out, int w, int h) {
    int idx = blockDim.x * blockIdx.x + threadIdx.x;
    int idy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetx = blockDim.x * gridDim.x;
    int offsety = blockDim.y * gridDim.y;
    double Gx = 0.0;
    double Gy = 0.0;
    uchar4 z1, z2, z3, z4, z5, z6, z7, z8;
    double w1, w2, w3, w4, w5, w6, w7, w8;

    for(long int a = idy; a < h; a += offsety) {
        for(long int b = idx; b < w; b += offsetx) {

            z1 = tex2D< uchar4 >(tex, b - 1, a - 1);
            w1 = grey(z1);
            z2 = tex2D< uchar4 >(tex, b, a - 1);
            w2 = grey(z2);
            z3 = tex2D< uchar4 >(tex, b + 1, a - 1);
            w3 = grey(z3);
            z4 = tex2D< uchar4 >(tex, b - 1, a);
            w4 = grey(z4);

            z5 = tex2D< uchar4 >(tex, b + 1, a);
            w5 = grey(z5);
            z6 = tex2D< uchar4 >(tex, b - 1, a + 1);
            w6 = grey(z6);
            z7 = tex2D< uchar4 >(tex, b, a + 1);
            w7 = grey(z7);
            z8 = tex2D< uchar4 >(tex, b + 1, a + 1);
            w8 = grey(z8);

            Gx = w3 + w5 + w5 + w8 - w1 - w4 - w4 - w6;
            Gy = w6 + w7 + w7 + w8 - w1 - w2 - w2 - w3;

            int GRAD = Gradient(Gx, Gy);
			int off = a * w + b;
            out[off].x = GRAD;
            out[off].y = GRAD;
            out[off].z = GRAD;
            out[off].w = 0;
        }
    }
}

int main() {
    char in_check[500];
    fgets(in_check, 500, stdin);
    in_check[strlen(in_check) - 1] = 0;
    char *in_name = in_check;

    char out_check[500];
    fgets(out_check, 500, stdin);
    out_check[strlen(out_check) - 1] = 0;
    char *out_name = out_check;

    int w, h;
    FILE *file = fopen(in_name, "rb");
    fread(&w, sizeof(int), 1, file);
    fread(&h, sizeof(int), 1, file);
    uchar4 *data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, file);
    fclose(file);

    cudaArray *arr;

    cudaChannelFormatDesc ch = cudaCreateChannelDesc<uchar4>();
	CSC(cudaMallocArray(&arr, &ch, w, h));
    CSC(cudaMemcpy2DToArray(arr, 0, 0, data, w * sizeof(uchar4), w * sizeof(uchar4), h, cudaMemcpyHostToDevice));

    struct cudaResourceDesc resDesc;
    memset(&resDesc, 0, sizeof(resDesc));
    resDesc.resType = cudaResourceTypeArray;
    resDesc.res.array.array = arr;

    struct cudaTextureDesc texDesc;
    memset(&texDesc, 0, sizeof(texDesc));
    texDesc.addressMode[0] = cudaAddressModeClamp;
    texDesc.addressMode[1] = cudaAddressModeClamp;
    texDesc.filterMode = cudaFilterModePoint;
    texDesc.readMode = cudaReadModeElementType;
    texDesc.normalizedCoords = false;

    cudaTextureObject_t tex = 0;
    CSC(cudaCreateTextureObject(&tex, &resDesc, &texDesc, NULL));

    int size = sizeof(uchar4) * w * h;

    uchar4 *dev_out;
    CSC(cudaMalloc(&dev_out, size));

    kernel<<< dim3(16, 16), dim3(16, 32) >>>(tex, dev_out, w, h);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, size, cudaMemcpyDeviceToHost));
    CSC(cudaDestroyTextureObject(tex));
    CSC(cudaFreeArray(arr));
    CSC(cudaFree(dev_out));

    file = fopen(out_name, "wb");
    fwrite(&w, sizeof(int), 1, file);
    fwrite(&h, sizeof(int), 1, file);
    fwrite(data, sizeof(uchar4), w * h, file);
    fclose(file);

    free(data);
    return 0;
}