#include <stdlib.h>
#include <math.h>
#include <stdio.h>

/*
* A pointer to this struct is returned if you call
* EigenResult* compute_eigenvalues(double** matrix, int n)
*/
typedef struct {
    double* values;     // Eigenvalues (sorted in ascending order)
    double** vectors;   // Eigenvectors stored as rows
    int n_eigs;         // Number of eigenvalues found
    int matrix_size;    // Size of original matrix
} EigenResult;

// LAPACK function declarations
extern "C" {
    // Symmetric eigenvalue problem
    void dsyevd_(char* jobz, char* uplo, int* n, double* a, int* lda,
                 double* w, double* work, int* lwork, int* iwork,
                 int* liwork, int* info);

    // Generalized symmetric eigenvalue problem (for ultrasoft PP)
    // Solves A*x = lambda*B*x where both A and B are symmetric
    void dsygvd_(int* itype, char* jobz, char* uplo, int* n,
                 double* a, int* lda, double* b, int* ldb,
                 double* w, double* work, int* lwork, int* iwork,
                 int* liwork, int* info);

    // Non-symmetric
    void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda,
                double* wr, double* wi, double* vl, int* ldvl,
                double* vr, int* ldvr, double* work, int* lwork, int* info);
}

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

// Main algorithm of my file
EigenResult* compute_eigenvalues(double** matrix, int n) {
    EigenResult* result = (EigenResult*)malloc(sizeof(EigenResult));
    result->n_eigs = n;
    result->matrix_size = n;
    result->values = (double*)malloc(n * sizeof(double));
    result->vectors = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        result->vectors[i] = (double*)malloc(n * sizeof(double));
    }
    
    // Need column-major format for LAPACK
    double* A = (double*)malloc(n * n * sizeof(double));
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[j * n + i] = matrix[i][j];
        }
    }

    if (is_symmetric(matrix, n)) {
        // Variables needed for LAPACK
        char jobz = 'V';
        char uplo = 'U';
        int info;
        int lwork = -1;
        int liwork = -1;
        double work_query;
        int iwork_query;
        
        dsyevd_(&jobz, &uplo, &n, A, &n, result->values, 
                &work_query, &lwork, &iwork_query, &liwork, &info);
        
        // Allocate workspace
        lwork = (int)work_query;
        liwork = iwork_query;
        double* work = (double*)malloc(lwork * sizeof(double));
        int* iwork = (int*)malloc(liwork * sizeof(int));
        
        // Actual computation
        dsyevd_(&jobz, &uplo, &n, A, &n, result->values, 
                work, &lwork, iwork, &liwork, &info);
        
        if (info != 0) {
            fprintf(stderr, "LAPACK dsyevd failed with info=%d\n", info);
            free(A); free(work); free(iwork);
            return result;
        }
        
        // Extract eigenvectors (LAPACK stores them in columns of A)
        for (int i = 0; i < n; i++) {
            for (int j = 0; j < n; j++) {
                result->vectors[i][j] = A[i * n + j];  // Column i of A
            }
        }
        
        free(work);
        free(iwork);
        
    } else {
        double* wr = (double*)malloc(n * sizeof(double));  // Real parts
        double* wi = (double*)malloc(n * sizeof(double));  // Imaginary parts
        double* vl = (double*)malloc(n * n * sizeof(double));  // Left eigenvectors (unused)
        double* vr = (double*)malloc(n * n * sizeof(double));  // Right eigenvectors
        
        // Again variables needed for LAPACK
        char jobvl = 'N';
        char jobvr = 'V';
        int info;
        int lwork = -1;
        double work_query;
        
        dgeev_(&jobvl, &jobvr, &n, A, &n, wr, wi, vl, &n, vr, &n, 
               &work_query, &lwork, &info);
        
        lwork = (int)work_query;
        double* work = (double*)malloc(lwork * sizeof(double));
        
        // Again actual computation
        dgeev_(&jobvl, &jobvr, &n, A, &n, wr, wi, vl, &n, vr, &n, 
               work, &lwork, &info);
        
        if (info != 0) {
            fprintf(stderr, "LAPACK dgeev failed with info=%d\n", info);
            free(A); free(wr); free(wi); free(vl); free(vr); free(work);
            return result;
        }
        
        // Sort eigenvalues by real part (ascending order)
        int* idx = (int*)malloc(n * sizeof(int));
        for (int i = 0; i < n; i++) idx[i] = i;
        
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
                fprintf(stderr, "Warning: Eigenvalue %d has imaginary part %.2e\n", 
                        i, wi[orig]);
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

/**
 * Solve generalized eigenvalue problem: H|ψ⟩ = E S|ψ⟩
 * Required for ultrasoft pseudopotentials where S ≠ I
 *
 * Uses LAPACK's dsygvd to solve A*x = lambda*B*x
 * where A = Hamiltonian, B = S-overlap matrix
 *
 * @param H Hamiltonian matrix (npw x npw)
 * @param S Overlap matrix (npw x npw)
 * @param n Matrix size (number of plane waves)
 * @return EigenResult containing eigenvalues and eigenvectors
 */
EigenResult* compute_generalized_eigenvalues(double** H, double** S, int n) {
    EigenResult* result = (EigenResult*)malloc(sizeof(EigenResult));
    result->n_eigs = n;
    result->matrix_size = n;
    result->values = (double*)malloc(n * sizeof(double));
    result->vectors = (double**)malloc(n * sizeof(double*));
    for (int i = 0; i < n; i++) {
        result->vectors[i] = (double*)malloc(n * sizeof(double));
    }

    // Convert to column-major format for LAPACK
    double* A = (double*)malloc(n * n * sizeof(double));  // Hamiltonian
    double* B = (double*)malloc(n * n * sizeof(double));  // Overlap matrix

    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            A[j * n + i] = H[i][j];
            B[j * n + i] = S[i][j];
        }
    }

    // LAPACK parameters for generalized symmetric eigenvalue problem
    int itype = 1;      // Problem type: A*x = lambda*B*x
    char jobz = 'V';    // Compute eigenvalues and eigenvectors
    char uplo = 'U';    // Upper triangle of A and B are stored
    int info;
    int lwork = -1;     // Query optimal workspace size
    int liwork = -1;
    double work_query;
    int iwork_query;

    // Workspace query
    dsygvd_(&itype, &jobz, &uplo, &n, A, &n, B, &n, result->values,
            &work_query, &lwork, &iwork_query, &liwork, &info);

    // Allocate workspace
    lwork = (int)work_query;
    liwork = iwork_query;
    double* work = (double*)malloc(lwork * sizeof(double));
    int* iwork = (int*)malloc(liwork * sizeof(int));

    // Actual computation
    dsygvd_(&itype, &jobz, &uplo, &n, A, &n, B, &n, result->values,
            work, &lwork, iwork, &liwork, &info);

    if (info != 0) {
        if (info > 0 && info <= n) {
            fprintf(stderr, "ERROR: dsygvd failed - B matrix is not positive definite (info=%d)\n", info);
            fprintf(stderr, "       Leading minor of order %d is not positive definite.\n", info);
        } else if (info > n) {
            fprintf(stderr, "ERROR: dsygvd failed - convergence failure (info=%d)\n", info);
        } else {
            fprintf(stderr, "ERROR: dsygvd failed with info=%d\n", info);
        }
        free(A); free(B); free(work); free(iwork);
        return result;
    }

    // Extract eigenvectors (LAPACK stores them in columns of A)
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            result->vectors[i][j] = A[i * n + j];  // Column i of A
        }
    }

    free(A);
    free(B);
    free(work);
    free(iwork);

    return result;
}