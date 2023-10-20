#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <iostream>
using namespace std;

#define CSC(call)  													              \
do {																                      \
	cudaError_t res = call;											            \
	if (res != cudaSuccess) {										            \
		fprintf(stderr, "ERROR in %s:%d. Message: %s\n",			\
				__FILE__, __LINE__, cudaGetErrorString(res));		  \
		exit(0);													                    \
	}																                        \
} while(0)

typedef struct {
    int a;
    int b;
} vector_1;

typedef struct {
    double x;
    double y;
    double z;
} vector_2;

__constant__ vector_2 avg[32];
__constant__ vector_2 normis_avg[32];

vector_2 new_vec_avg[32];
vector_2 new_normis_avg[32];

__device__ __host__ void Get(uchar4* data, vector_2& kek) {
    kek.x = data->x;
    kek.y = data->y;
    kek.z = data->z;
}

__device__ double find_pixel(uchar4 check, int number) {
    double result = 0;
    vector_2 kek;
    Get(&check, kek);

    double new_RGB[3];
    double new_norm[3];

    new_RGB[0] = kek.x;
    new_RGB[1] = kek.y;
    new_RGB[2] = kek.z;

    new_norm[0] = normis_avg[number].x;
    new_norm[1] = normis_avg[number].y;
    new_norm[2] = normis_avg[number].z;
    for (int i = 0; i < 3; i++) {
        result += new_RGB[i] * new_norm[i];
    }
    return result;
}

__global__ void kernel(uchar4* data, int w, int h, int nc) {
    int xdx = blockDim.x * blockIdx.x + threadIdx.x;
    int ydy = blockDim.y * blockIdx.y + threadIdx.y;
    int offsetX = blockDim.x * gridDim.x;
    int offsetY = blockDim.y * gridDim.y;
    uchar4 check;

    for (int i = ydy; i < h; i += offsetY) {
        for (int j = xdx; j < w; j += offsetX) {
            check = data[i * w + j];
            double finded_1 = find_pixel(check, 0);
            int n = 0;
            for (int a = 1; a < nc; a++) {
                double finded_2 = find_pixel(check, a);
                if (finded_1 < finded_2) {
                    finded_1 = finded_2;
                    n = a;
                }
            }
            data[i * w + j].w = n;
        }
    }
}

void vec_avg_find(vector<vector<vector_1>>& vector_new, uchar4* data, int w, int h, int nc) {
    vector<vector_2> cpu(32);
    for (int a = 0; a < nc; a++) {
        cpu[a].x = 0;
        cpu[a].y = 0;
        cpu[a].z = 0;
        for (int b = 0; b < vector_new[a].size(); b++) {
            vector_1 n = vector_new[a][b];
            uchar4 check = data[n.b * w + n.a];
            vector_2 kek;
            Get(&check, kek);

            cpu[a].x += kek.x;
            cpu[a].y += kek.y;
            cpu[a].z += kek.z;
        }
        cpu[a].x = cpu[a].x / vector_new[a].size();
        cpu[a].y = cpu[a].y / vector_new[a].size();
        cpu[a].z = cpu[a].z / vector_new[a].size();
    }
    for (int x = 0; x < nc; x++) {
        new_vec_avg[x] = cpu[x];
    }
}

void normis_avg_find(int nc) {
    for (int i = 0; i < nc; i++) {
        new_normis_avg[i].x = (double)new_vec_avg[i].x / pow((new_vec_avg[i].x * new_vec_avg[i].x + new_vec_avg[i].y * new_vec_avg[i].y + new_vec_avg[i].z * new_vec_avg[i].z), 0.5);
        new_normis_avg[i].y = (double)new_vec_avg[i].y / pow((new_vec_avg[i].x * new_vec_avg[i].x + new_vec_avg[i].y * new_vec_avg[i].y + new_vec_avg[i].z * new_vec_avg[i].z), 0.5);
        new_normis_avg[i].z = (double)new_vec_avg[i].z / pow((new_vec_avg[i].x * new_vec_avg[i].x + new_vec_avg[i].y * new_vec_avg[i].y + new_vec_avg[i].z * new_vec_avg[i].z), 0.5);
    }
}

int main() {
    int w, h, nc, np;

    char in_check[500];
    fgets(in_check, 500, stdin);
    in_check[strlen(in_check) - 1] = 0;
    char *in_name = in_check;

    char out_check[500];
    fgets(out_check, 500, stdin);
    out_check[strlen(out_check) - 1] = 0;
    char *out_name = out_check;

    FILE* file = fopen(in_name, "rb");
    fread(&w, sizeof(int), 1, file);
    fread(&h, sizeof(int), 1, file);
    uchar4* data = (uchar4*)malloc(sizeof(uchar4) * w * h);
    fread(data, sizeof(uchar4), w * h, file);
    fclose(file);

    scanf("%d", &nc);
    vector<vector<vector_1>> vector_new(nc);
    for (int i = 0; i < nc; i++) {
        scanf("%d", &np);
        vector_new[i].resize(np);
        for (int j = 0; j < np; j++) {
            scanf("%d", &vector_new[i][j].a);
            scanf("%d", &vector_new[i][j].b);
        }
    }

    vec_avg_find(vector_new, data, w, h, nc);
    normis_avg_find(nc);

    CSC(cudaMemcpyToSymbol(avg, new_vec_avg, 32 * sizeof(vector_2)));
    CSC(cudaMemcpyToSymbol(normis_avg, new_normis_avg, 32 * sizeof(vector_2)));

    uchar4* dev_out;
    int size = sizeof(uchar4) * w * h;
    CSC(cudaMalloc(&dev_out, size));
    CSC(cudaMemcpy(dev_out, data, size, cudaMemcpyHostToDevice));

    kernel<<< dim3(16, 16), dim3(16, 32) >>>(dev_out, w, h, nc);
    CSC(cudaGetLastError());

    CSC(cudaMemcpy(data, dev_out, size, cudaMemcpyDeviceToHost));

    CSC(cudaFree(dev_out));

    file = fopen(out_name, "wb");
    fwrite(&w, sizeof(int), 1, file);
    fwrite(&h, sizeof(int), 1, file);
    fwrite(data, sizeof(uchar4), w * h, file);
    fclose(file);

    free(data);
    return 0;
}