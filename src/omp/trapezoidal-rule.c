#include <omp.h>
#include <stdio.h>

// Function to be integrated
double f(double x);

double TrapezoidalRule(double x0, double x1, int n_trapezoids);

int main() {
    // Integration domain and number of trapezoids
    double x0 = 0.0;
    double x1 = 4.0;
    int n_trapezoids = 500000000;
    int n_threads = 4;
    double start, end, runtime_serial, runtime_parallel, result, speedup;

    // Run serial version
    omp_set_num_threads(1);

    start = omp_get_wtime();
    result = TrapezoidalRule(x0, x1, n_trapezoids);
    end = omp_get_wtime();

    runtime_serial = end - start;
    printf("   Serial Result: %.10f\n", result);
    printf("  Serial Runtime: %f\n", runtime_serial);

    // Run parallel version
    omp_set_num_threads(n_threads);

    start = omp_get_wtime();
    result = TrapezoidalRule(x0, x1, n_trapezoids);
    end = omp_get_wtime();

    runtime_parallel = end - start;
    printf(" Parallel Result: %.10f\n", result);
    printf("Parallel Runtime: %f\n", runtime_parallel);

    // Calculate speedup
    speedup = runtime_serial / runtime_parallel;
    printf("         Speedup: %f x", speedup);

    return 0;
}

double TrapezoidalRule(double x0, double x1, int n_trapezoids) {

    double h = (x1 - x0) / n_trapezoids;
    double result = (f(x0) + f(x1)) / 2.0;
    double x, local_result;
    int i;

#pragma omp parallel for default(none) reduction(+:result)                     \
    private(i, x, local_result) shared(n_trapezoids, x0, h)
    for (i = 1; i < n_trapezoids; i++) {
        x = x0 + i * h;
        local_result = f(x);
        result += local_result;
    }

    result *= h;
    return result;
}

double f(double x) { return x; }
