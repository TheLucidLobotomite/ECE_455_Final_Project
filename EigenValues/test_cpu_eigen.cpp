#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <chrono>

// ============= MATCH SOLVER'S STRUCT =============
typedef struct {
    double* values;
    double** vectors;
    int n_eigs;
    int matrix_size;
} EigenResult;

// Forward declarations
EigenResult* compute_eigenvalues(double** matrix, int n);
void free_eigen_result(EigenResult* result);

// ============= TEST MATRICES =============

// Create sparse Hamiltonian (tridiagonal + random sparse elements)
double** create_sparse_hamiltonian(int n, double sparsity, unsigned int seed) {
    srand(seed);
    double** H = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        H[i] = (double*)calloc(n, sizeof(double));
    }
    
    // Diagonal: on-site energies
    for (int i = 0; i < n; i++) {
        H[i][i] = 1.0 + ((double)rand() / RAND_MAX) * 4.0;
    }
    
    // Off-diagonal: nearest neighbor hopping (symmetric)
    for (int i = 0; i < n-1; i++) {
        double val = -0.1 - ((double)rand() / RAND_MAX) * 0.9;
        H[i][i+1] = val;
        H[i+1][i] = val;
    }
    
    // Add sparse random interactions (symmetric)
    int n_random = (int)(n * n * sparsity);
    for (int r = 0; r < n_random; r++) {
        int i = rand() % n;
        int j = rand() % n;
        if (i != j) {
            double val = ((double)rand() / RAND_MAX - 0.5) * 1.0;
            H[i][j] = val;
            H[j][i] = val;  // Ensure symmetry
        }
    }
    
    return H;
}

// ============= UTILITIES =============

void free_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++) free(mat[i]);
    free(mat);
}

// ============= TEST RUNNER =============

int main() {
    printf("\n=== LAPACK EIGENVALUE SOLVER BENCHMARK ===\n\n");

    const int n = 500;
    const int num_runs = 100;
    const double sparsity = 0.01;
    double* times_ms = (double*)malloc(num_runs * sizeof(double));

    printf("Running %d iterations of %dx%d sparse Hamiltonian eigensolve...\n\n", num_runs, n, n);

    for (int run = 0; run < num_runs; run++) {
        // Create a new random matrix for each run
        unsigned int seed = 42 + run;
        double** A = create_sparse_hamiltonian(n, sparsity, seed);

        // Time the eigenvalue computation
        auto t0 = std::chrono::high_resolution_clock::now();
        EigenResult* r = compute_eigenvalues(A, n);
        auto t1 = std::chrono::high_resolution_clock::now();

        double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
        times_ms[run] = ms;

        //printf("Run %2d: %.6f ms\n", run + 1, ms);

        // Cleanup
        free_eigen_result(r);
        free_matrix(A, n);
    }

    // Calculate statistics
    double total_time = 0.0;
    double min_time = times_ms[0];
    double max_time = times_ms[0];
    
    for (int i = 0; i < num_runs; i++) {
        total_time += times_ms[i];
        if (times_ms[i] < min_time) min_time = times_ms[i];
        if (times_ms[i] > max_time) max_time = times_ms[i];
    }
    
    double avg_time = total_time / num_runs;

    printf("\n");
    printf("======================================================================\n");
    printf("BENCHMARK RESULTS\n");
    printf("======================================================================\n");
    printf("Total runs:    %d\n", num_runs);
    printf("Average time:  %.6f ms\n", avg_time);
    printf("Min time:      %.6f ms\n", min_time);
    printf("Max time:      %.6f ms\n", max_time);
    printf("\n");

    /*
    printf("All times (ms):\n[");
    for (int i = 0; i < num_runs; i++) {
        printf("%.6f", times_ms[i]);
        if (i < num_runs - 1) printf(", ");
    }
    printf("]\n\n");

    free(times_ms);
    */
   
    return 0;
}