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
EigenResult* davidson_algorithm(double** H, int n, int k, int max_iter, double tol);
void free_eigen_result(EigenResult* result);

// ============= TEST MATRICES =============

double** create_2x2_matrix() {
    double** A = (double**)malloc(2 * sizeof(double*));
    for (int i = 0; i < 2; i++) A[i] = (double*)malloc(2 * sizeof(double));
    A[0][0] = 3; A[0][1] = 1;
    A[1][0] = 0; A[1][1] = 2;
    return A;
}

double** create_3x3_matrix() {
    double** A = (double**)malloc(3 * sizeof(double*));
    for (int i = 0; i < 3; i++) A[i] = (double*)malloc(3 * sizeof(double));
    A[0][0] = 2;  A[0][1] = 0;  A[0][2] = 1;
    A[1][0] = -1; A[1][1] = 4;  A[1][2] = -1;
    A[2][0] = -1; A[2][1] = 2;  A[2][2] = 0;
    return A;
}

double** create_4x4_matrix() {
    double** A = (double**)malloc(4 * sizeof(double*));
    for (int i = 0; i < 4; i++) A[i] = (double*)malloc(4 * sizeof(double));
    A[0][0] = 4; A[0][1] = -1; A[0][2] = 0;  A[0][3] = 0;
    A[1][0] = -1; A[1][1] = 4; A[1][2] = -1; A[1][3] = 0;
    A[2][0] = 0; A[2][1] = -1; A[2][2] = 4;  A[2][3] = -1;
    A[3][0] = 0; A[3][1] = 0;  A[3][2] = -1; A[3][3] = 4;
    return A;
}

double** create_5x5_matrix() {
    double** A = (double**)malloc(5 * sizeof(double*));
    for (int i = 0; i < 5; i++) A[i] = (double*)malloc(5 * sizeof(double));
    A[0][0] = 5; A[0][1] = 1; A[0][2] = 0; A[0][3] = 0; A[0][4] = 0;
    A[1][0] = 1; A[1][1] = 4; A[1][2] = 1; A[1][3] = 0; A[1][4] = 0;
    A[2][0] = 0; A[2][1] = 1; A[2][2] = 3; A[2][3] = 1; A[2][4] = 0;
    A[3][0] = 0; A[3][1] = 0; A[3][2] = 1; A[3][3] = 2; A[3][4] = 1;
    A[4][0] = 0; A[4][1] = 0; A[4][2] = 0; A[4][3] = 1; A[4][4] = 1;
    return A;
}

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
    
    // Off-diagonal: nearest neighbor hopping
    for (int i = 0; i < n-1; i++) {
        double val = -0.1 - ((double)rand() / RAND_MAX) * 0.9;
        H[i][i+1] = val;
        H[i+1][i] = val;
    }
    
    // Add sparse random interactions
    int n_random = (int)(n * n * sparsity);
    for (int r = 0; r < n_random; r++) {
        int i = rand() % n;
        int j = rand() % n;
        if (i != j) {
            double val = ((double)rand() / RAND_MAX - 0.5) * 1.0;
            H[i][j] = val;
            H[j][i] = val;
        }
    }
    
    return H;
}

// ============= UTILITIES =============

void free_matrix(double** mat, int n) {
    for (int i = 0; i < n; i++) free(mat[i]);
    free(mat);
}

void print_matrix(double** A, int n, const char* name) {
    printf("%s Matrix:\n", name);
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++)
            printf("%8.4f ", A[i][j]);
        printf("\n");
    }
    printf("\n");
}

void print_eigenvalues(EigenResult* r, int k) {
    printf("Eigenvalues:\n");
    for (int i = 0; i < k; i++)
        printf("  lambda_%d = %.10f\n", i, r->values[i]);
    printf("\n");
}

void print_eigenvectors(EigenResult* r, int max_rows) {
    printf("Eigenvectors (first few rows):\n");
    int n = r->matrix_size;
    int k = r->n_eigs;
    int rows = (n < max_rows ? n : max_rows);

    for (int i = 0; i < rows; i++) {
        printf("  [");
        for (int j = 0; j < k; j++) {
            printf("%9.6f", r->vectors[j][i]);
            if (j < k-1) printf(", ");
        }
        printf("]\n");
    }
    printf("\n");
}

void verify_eigenpair(double** A, int n, double lambda, double* v, const char* name) {
    double* Av = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) {
        Av[i] = 0.0;
        for (int j = 0; j < n; j++) {
            Av[i] += A[i][j] * v[j];
        }
    }
    
    double residual = 0.0;
    for (int i = 0; i < n; i++) {
        double diff = Av[i] - lambda * v[i];
        residual += diff * diff;
    }
    residual = sqrt(residual);
    
    printf("  %s: ||A*v - lambda*v|| = %.2e %s\n", 
           name, residual, (residual < 1e-8) ? "OK" : "FAIL");
    
    free(Av);
}

// ============= TEST RUNNER =============

void test_small_matrix(const char* name, double** (*maker)(), int n, int k) {
    printf("======================================================================\n");
    printf("%s Matrix Test\n", name);
    printf("======================================================================\n");

    double** A = maker();
    print_matrix(A, n, name);

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenResult* r = davidson_algorithm(A, n, k, 250, 1e-8);
    auto t1 = std::chrono::high_resolution_clock::now();

    print_eigenvalues(r, k);
    print_eigenvectors(r, 4);

    printf("Verification (||A*v - lambda*v||):\n");
    for (int i = 0; i < k; i++) {
        char label[32];
        sprintf(label, "lambda_%d", i);
        verify_eigenpair(A, n, r->values[i], r->vectors[i], label);
    }
    printf("\n");

    double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
    printf("Computation time: %.6f ms\n\n", ms);

    free_eigen_result(r);
    free_matrix(A, n);
}

void test_large_matrix(const char* name, int n, int k, double sparsity, unsigned int seed) {
    printf("======================================================================\n");
    printf("%s Matrix Test\n", name);
    printf("======================================================================\n");

    printf("Matrix size: %dx%d (sparse)\n", n, n);
    printf("Finding %d smallest eigenvalues...\n\n", k);

    double** A = create_sparse_hamiltonian(n, sparsity, seed);

    auto t0 = std::chrono::high_resolution_clock::now();
    EigenResult* r = davidson_algorithm(A, n, k, 250, 1e-8);
    auto t1 = std::chrono::high_resolution_clock::now();

    print_eigenvalues(r, k);

    printf("Verification (||A*v - lambda*v||):\n");
    for (int i = 0; i < k; i++) {
        char label[32];
        sprintf(label, "lambda_%d", i);
        verify_eigenpair(A, n, r->values[i], r->vectors[i], label);
    }
    printf("\n");

    double ms = std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count() / 1e6;
    printf("Computation time: %.6f ms (%.6f seconds)\n\n", ms, ms/1000.0);

    free_eigen_result(r);
    free_matrix(A, n);
}

int main() {
    printf("\n=== EIGENVALUE SOLVER TESTS ===\n\n");

    // Small tests
    printf("SMALL MATRIX TESTS\n");
    printf("==================\n\n");
    test_small_matrix("2x2", create_2x2_matrix, 2, 2);
    test_small_matrix("3x3", create_3x3_matrix, 3, 3);
    test_small_matrix("4x4", create_4x4_matrix, 4, 4);
    test_small_matrix("5x5", create_5x5_matrix, 5, 5);

    // Large sparse tests
    printf("\n\nLARGE SPARSE HAMILTONIAN TESTS\n");
    printf("===============================\n\n");
    test_large_matrix("1000x1000 Sparse Hamiltonian (k=5)", 1000, 5, 0.001, 42);
    test_large_matrix("10000x10000 Sparse Hamiltonian (k=5)", 10000, 5, 0.0001, 42);

    // This whole tester is on 1000 for max iterations and 1e-8 tolerance
    return 0;
}