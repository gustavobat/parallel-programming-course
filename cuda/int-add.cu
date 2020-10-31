#include <iostream>

// Macro for checking cuda errors following a cuda launch or api call
static void checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_ERROR(err) (checkCudaErrors(err, __FILE__, __LINE__))

__global__ void intAddKernel(const int *a, const int *b, int *c) {
    *c = *a + *b;
}

int main() {
    int h_A = 2;
    int h_B = 2;
    int h_C;

    int *d_A, *d_B, *d_C;

    CHECK_ERROR(cudaMalloc((void**)&d_A, sizeof(int)));
    CHECK_ERROR(cudaMalloc((void**)&d_B, sizeof(int)));
    CHECK_ERROR(cudaMalloc((void**)&d_C, sizeof(int)));

    CHECK_ERROR(cudaMemcpy(d_A, &h_A, sizeof(int), cudaMemcpyHostToDevice));
    CHECK_ERROR(cudaMemcpy(d_B, &h_B, sizeof(int), cudaMemcpyHostToDevice));
    intAddKernel<<<1, 1>>>(d_A, d_B, d_C);
    CHECK_ERROR(cudaMemcpy(&h_C, d_C, sizeof(int), cudaMemcpyDeviceToHost));

    CHECK_ERROR(cudaFree(d_A));
    CHECK_ERROR(cudaFree(d_B));
    CHECK_ERROR(cudaFree(d_C));

    printf("%d + %d = %d!\n", h_A, h_B, h_C);
    return 0;
}
