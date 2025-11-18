#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    double* values;     // Eigenvalues (sorted in ascending order)
    double** vectors;   // Eigenvectors stored as columns (vector i is vectors[i][j] for j=0..n-1)
    int n_eigs;         // Number of eigenvalues found
    int matrix_size;    // Size of original matrix
} EigenResult;

// LAPACK function declarations
extern "C" {
    // Symmetric matrix eigensolver
    void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
                double* w, double* work, int* lwork, int* info);
    
    // General matrix eigensolver
    void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda,
                double* wr, double* wi, double* vl, int* ldvl,
                double* vr, int* ldvr, double* work, int* lwork, int* info);
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
 * Compute all eigenvalues and eigenvectors of a matrix using LAPACK
 * 
 * @param matrix Input matrix (row-major format: matrix[row][col])
 * @param n Size of the matrix (n x n)
 * @return EigenResult structure containing eigenvalues and eigenvectors
 * 
 * LAPACK Data Format Notes:
 * -------------------------
 * 
 * For SYMMETRIC matrices (dsyev):
 * - Input: Upper or lower triangle of the matrix
 * - Output eigenvalues (w): Sorted in ASCENDING order
 * - Output eigenvectors: Stored in COLUMNS of matrix A
 *   - Eigenvector i corresponds to eigenvalue w[i]
 *   - Column j is A[j*n + i] for i = 0..n-1
 * 
 * For GENERAL matrices (dgeev):
 * - Output eigenvalues: wr (real part) + wi*i (imaginary part)
 * - Output eigenvectors: Stored in COLUMNS of vr matrix
 *   - If wi[j] == 0: eigenvector j is vr[j*n + i] for i = 0..n-1
 *   - If wi[j] != 0: complex conjugate pair (we only use real part)
 * - Eigenvalues are NOT sorted by default
 */
EigenResult* compute_eigenvalues(double** matrix, int n) {
    // Allocate result structure
    EigenResult* result = (EigenResult*)malloc(sizeof(EigenResult));
    result->n_eigs = n;
    result->matrix_size = n;
    result->values = (double*)malloc(n * sizeof(double));
    result->vectors = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        result->vectors[i] = (double*)malloc(n * sizeof(double));
    }
    
    // Convert matrix from row-major to column-major format for LAPACK
    double* A = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[j * n + i] = matrix[i][j];
        }
    }
    
    // Dsyev is faster and more accurate for symmetric matrices
    if (is_symmetric(matrix, n)) {
        // Allocate workspace for eigenvalues
        double* w = (double*)malloc(n * sizeof(double));
        
        // LAPACK parameters
        char jobz = 'V';  // Compute both eigenvalues and eigenvectors
        char uplo = 'U';  // Upper triangle of A is stored
        int lwork = -1;   // Workspace query
        double work_query;
        int info;
        
        // Query optimal workspace size
        dsyev_(&jobz, &uplo, &n, A, &n, w, &work_query, &lwork, &info);
        lwork = (int)work_query;
        double* work = (double*)malloc(lwork * sizeof(double));
        
        // Compute eigenvalues and eigenvectors
        dsyev_(&jobz, &uplo, &n, A, &n, w, work, &lwork, &info);
        
        if (info != 0) {
            fprintf(stderr, "LAPACK dsyev failed with info=%d\n", info);
            free(A); free(w); free(work);
            return result;
        }
        
        // Extract results (already sorted in ascending order)
        // LAPACK returns eigenvectors in columns of A
        for (int i = 0; i < n; i++) {
            result->values[i] = w[i];
            for (int j = 0; j < n; j++) {
                result->vectors[i][j] = A[i * n + j];  // Column i of A
            }
        }
        
        free(w);
        free(work);
    } else {
        // Allocate workspace for eigenvalues and eigenvectors
        double* wr = (double*)malloc(n * sizeof(double));  // Real parts
        double* wi = (double*)malloc(n * sizeof(double));  // Imaginary parts
        double* vl = (double*)malloc(n * n * sizeof(double));  // Left eigenvectors (not used)
        double* vr = (double*)malloc(n * n * sizeof(double));  // Right eigenvectors
        
        // LAPACK parameters
        char jobvl = 'N';  // Don't compute left eigenvectors
        char jobvr = 'V';  // Compute right eigenvectors
        int lwork = -1;    // Workspace query
        double work_query;
        int info;
        
        // Query optimal workspace size
        dgeev_(&jobvl, &jobvr, &n, A, &n, wr, wi, vl, &n, vr, &n, &work_query, &lwork, &info);
        lwork = (int)work_query;
        double* work = (double*)malloc(lwork * sizeof(double));
        
        // Compute eigenvalues and eigenvectors
        dgeev_(&jobvl, &jobvr, &n, A, &n, wr, wi, vl, &n, vr, &n, work, &lwork, &info);
        
        if (info != 0) {
            fprintf(stderr, "LAPACK dgeev failed with info=%d\n", info);
            free(A); free(wr); free(wi); free(vl); free(vr); free(work);
            return result;
        }
        
        // Sort eigenvalues by real part (ascending order)
        int* idx = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) idx[i] = i;
        
        // Simple bubble sort
        for (int i = 0; i < n - 1; i++) {
            for (int j = i + 1; j < n; j++) {
                if (wr[idx[j]] < wr[idx[i]]) {
                    int tmp = idx[i];
                    idx[i] = idx[j];
                    idx[j] = tmp;
                }
            }
        }
        
        // Extract results (sorted by real part)
        for (int i = 0; i < n; i++) {
            int orig = idx[i];
            
            // Warn if eigenvalue has significant imaginary part
            if (fabs(wi[orig]) > 1e-10) {
                fprintf(stderr, "Warning: Eigenvalue %d has imaginary part %.2e\n", i, wi[orig]);
            }
            
            result->values[i] = wr[orig];
            
            // Extract eigenvector from column 'orig' of vr
            for (int j = 0; j < n; j++) {
                result->vectors[i][j] = vr[orig * n + j];
            }
        }
        
        free(wr); free(wi); free(vl); free(vr); free(work); free(idx);
    }
    
    free(A);
    return result;
}

void free_eigen_result(EigenResult* result) {
    if (result) {
        free(result->values);
        for (int i = 0; i < result->n_eigs; i++) {
            free(result->vectors[i]);
        }
        free(result->vectors);
        free(result);
    }
}