#include <omp.h>
#include <stdio.h>

// Function to be integrated
double f(double x);

double TrapezoidalRule(double x0, double x1, int n_trapezoids);

int main() {
    // Set number of threads
    int n_threads = 8;
    omp_set_num_threads(n_threads);

    // Integration domain and number of trapezoids
    double x0 = 0.0;
    double x1 = 4.0;
    int n_trapezoids = 500;
    
    double result = TrapezoidalRule(x0, x1, n_trapezoids);
    printf("The approximated result is %.10f\n", result);

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
