#include <omp.h>
#include <stdio.h>

void Hello();

int main() {
    int nthreads = 8;

#pragma omp parallel num_threads(nthreads) default(none)
    Hello();
}

void Hello() {
#ifdef _OPENMP
    int my_rank = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
#else
    int my_rank = 0;
    int nthreads = 1;
#endif
    printf("Hello from thread %d of %d!\n", my_rank, nthreads);
}
