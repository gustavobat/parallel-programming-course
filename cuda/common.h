//
// Created by Gustavo Batistela on 31/10/2020.
//
#include <iostream>

#ifndef CUDA_COMMON_H
#define CUDA_COMMON_H

// Macro for checking cuda errors following a cuda launch or api call
static void checkCudaErrors(cudaError_t err, const char *file, int line) {
    if (err != cudaSuccess) {
        printf("%s in %s at line %d\n", cudaGetErrorString(err), file, line);
        exit(EXIT_FAILURE);
    }
}
#define CHECK_ERROR(err) (checkCudaErrors(err, __FILE__, __LINE__))

#endif
