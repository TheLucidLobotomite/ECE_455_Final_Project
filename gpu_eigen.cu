#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

typedef struct {
    double* values;     // Eigenvalues (sorted in ascending order)
    double** vectors;   // Eigenvectors stored as columns (vector i is vectors[i][j] for j=0..n-1)
    int n_eigs;         // Number of eigenvalues found
    int matrix_size;    // Size of original matrix
} EigenResult;

// Forward declare internal CUDA eigensolver
EigenResult* compute_eigenvalues_gpu_impl(double** matrix, int n);

extern "C" {

// This is the function called by test_dft.cpp
EigenResult* compute_eigenvalues_gpu(double** matrix, int n) {
    return compute_eigenvalues_gpu_impl(matrix, n);
}

// This is the memory-freeing function test_dft.cpp expects
void free_eigen_result(EigenResult* result) {
    if (!result) return;

    free(result->values);

    for (int i = 0; i < result->n_eigs; i++)
        free(result->vectors[i]);

    free(result->vectors);
    free(result);
}

} // extern "C"

// ---------- CUDA Error Checking ----------
#define checkCuda(val) check_cuda((val), #val, __FILE__, __LINE__)
void check_cuda(cudaError_t err, const char* const func, const char* const file, int line) {
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA Error at %s:%d — %s (%s)\n", 
                file, line, cudaGetErrorString(err), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCusolver(val) check_cusolver((val), #val, __FILE__, __LINE__)
void check_cusolver(cusolverStatus_t err, const char* const func, const char* const file, int line) {
    if (err != CUSOLVER_STATUS_SUCCESS) {
        fprintf(stderr, "cuSOLVER Error at %s:%d — code %d (%s)\n", 
                file, line, err, func);
        exit(EXIT_FAILURE);
    }
}

// Check if matrix is symmetric
bool is_symmetric(double** matrix, int n, double tol = 1e-10) {
    for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
            if (fabs(matrix[i][j] - matrix[j][i]) > tol) {
                return false;
            }
        }
    }
    return true;
}

/**
 * Compute all eigenvalues and eigenvectors using cuSOLVER (GPU)
 * 
 * @param matrix Input matrix (row-major format: matrix[row][col])
 * @param n Size of the matrix (n x n)
 * @return EigenResult structure containing eigenvalues and eigenvectors
 * 
 * cuSOLVER Data Format Notes:
 * ---------------------------
 * cuSOLVER uses COLUMN-MAJOR format (like LAPACK/BLAS).
 * 
 * For SYMMETRIC matrices (cusolverDnDsyevd):
 * - Input: Full matrix (we use lower triangle)
 * - Uses divide-and-conquer algorithm (very fast)
 * - Output eigenvalues (W): Sorted in ASCENDING order
 * - Output eigenvectors: Stored in COLUMNS of matrix A (overwrites input)
 *   - Eigenvector i corresponds to eigenvalue W[i]
 *   - Column j is A[j*n + i] for i = 0..n-1
 * 
 * NOTE: This implementation only supports SYMMETRIC matrices.
 * For general matrices, you need CUDA 11+ with cusolverDnDgeev or use CPU LAPACK.
 */
EigenResult* compute_eigenvalues_gpu_impl(double** matrix, int n) {
    // Allocate result structure
    EigenResult* result = (EigenResult*)malloc(sizeof(EigenResult));
    result->n_eigs = n;
    result->matrix_size = n;
    result->values = (double*)malloc(n * sizeof(double));
    result->vectors = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        result->vectors[i] = (double*)malloc(n * sizeof(double));
    }
    
    // Check if symmetric
    bool symmetric = is_symmetric(matrix, n);
    
    if (!symmetric) {
        fprintf(stderr, "ERROR: Matrix is not symmetric!\n");
        fprintf(stderr, "cuSOLVER on CUDA 13.0 only supports symmetric matrices for eigenvalue problems.\n");
        fprintf(stderr, "For general matrices, please use the CPU LAPACK version.\n");
        exit(EXIT_FAILURE);
    }
    
    // Convert matrix from row-major to column-major format
    double* A_host = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A_host[j * n + i] = matrix[i][j];  // Transpose: row-major to column-major
        }
    }
    
    // Create cuSOLVER handle
    cusolverDnHandle_t cusolverH = NULL;
    checkCusolver(cusolverDnCreate(&cusolverH));
    
    // Allocate device memory
    double *d_A = NULL, *d_W = NULL;
    int *d_info = NULL;
    
    checkCuda(cudaMalloc((void**)&d_A, n * n * sizeof(double)));
    checkCuda(cudaMalloc((void**)&d_W, n * sizeof(double)));
    checkCuda(cudaMalloc((void**)&d_info, sizeof(int)));
    
    // Copy matrix to device
    checkCuda(cudaMemcpy(d_A, A_host, n * n * sizeof(double), cudaMemcpyHostToDevice));
    
    // Query workspace size
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;  // Compute eigenvectors
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_LOWER;     // Lower triangle
    int lwork = 0;
    checkCusolver(cusolverDnDsyevd_bufferSize(cusolverH, jobz, uplo, n, d_A, n, d_W, &lwork));
    
    // Allocate workspace
    double* d_work = NULL;
    checkCuda(cudaMalloc((void**)&d_work, lwork * sizeof(double)));
    
    // Compute eigenvalues and eigenvectors
    checkCusolver(cusolverDnDsyevd(cusolverH, jobz, uplo, n, d_A, n, d_W, d_work, lwork, d_info));
    
    // Synchronize to ensure computation is complete
    checkCuda(cudaDeviceSynchronize());
    
    // Check for convergence
    int info_host = 0;
    checkCuda(cudaMemcpy(&info_host, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info_host != 0) {
        fprintf(stderr, "cusolverDnDsyevd failed with info=%d\n", info_host);
    }
    
    // Copy results back to host
    checkCuda(cudaMemcpy(A_host, d_A, n * n * sizeof(double), cudaMemcpyDeviceToHost));
    checkCuda(cudaMemcpy(result->values, d_W, n * sizeof(double), cudaMemcpyDeviceToHost));
    
    // Extract eigenvectors from column-major format
    // Eigenvectors are stored in columns of A_host
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->vectors[i][j] = A_host[i * n + j];  // Column i of A_host
        }
    }
    
    // Cleanup
    checkCuda(cudaFree(d_A));
    checkCuda(cudaFree(d_W));
    checkCuda(cudaFree(d_info));
    checkCuda(cudaFree(d_work));
    checkCusolver(cusolverDnDestroy(cusolverH));
    free(A_host);
    
    return result;
}