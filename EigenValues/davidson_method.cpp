#include <stdlib.h>
#include <math.h>
#include <stdio.h>

typedef struct {
    double* values;     // Eigenvalues
    double** vectors;   // Eigenvectors (column-wise)
    int n_eigs;         // Number of eigenvalues found
    int matrix_size;    // Size of original matrix
} EigenResult;

// LAPACK function declarations
extern "C" {
    void dsyev_(char* jobz, char* uplo, int* n, double* a, int* lda,
                double* w, double* work, int* lwork, int* info);
    
    void dgeev_(char* jobvl, char* jobvr, int* n, double* a, int* lda,
                double* wr, double* wi, double* vl, int* ldvl,
                double* vr, int* ldvr, double* work, int* lwork, int* info);
}

// Linear algebra utilities
void matrix_vector_mult(double** A, double* v, double* result, int n) {
    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        for (int j = 0; j < n; j++) sum += A[i][j] * v[j];
        result[i] = sum;
    }
}

double vector_dot(const double* a, const double* b, int n) {
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += a[i] * b[i];
    return sum;
}

double vector_norm(const double* v, int n) {
    return sqrt(vector_dot(v, v, n));
}

double vector_normalize(double* v, int n) {
    double norm = vector_norm(v, n);
    if (norm < 1e-15) return 0.0;
    for (int i = 0; i < n; i++) v[i] /= norm;
    return norm;
}

void orthonormalize(double** V, int num, int n) {
    for (int i = 0; i < num; ++i) {
        for (int j = 0; j < i; ++j) {
            double dot = vector_dot(V[j], V[i], n);
            for (int l = 0; l < n; ++l) V[i][l] -= dot * V[j][l];
        }
        // Second pass for numerical stability
        for (int j = 0; j < i; ++j) {
            double dot = vector_dot(V[j], V[i], n);
            for (int l = 0; l < n; ++l) V[i][l] -= dot * V[j][l];
        }
        
        double norm = vector_normalize(V[i], n);
        if (norm < 1e-15) {
            for (int l = 0; l < n; ++l) V[i][l] = ((double)rand() / RAND_MAX) - 0.5;
            i--;
        }
    }
}

void orthonormalize_against(double** V, int num, double* v, int n) {
    int max_attempts = 3;
    for (int attempt = 0; attempt < max_attempts; ++attempt) {
        // Two passes of Gram-Schmidt for better orthogonality
        for (int pass = 0; pass < 2; ++pass) {
            for (int i = 0; i < num; ++i) {
                double dot = vector_dot(V[i], v, n);
                for (int l = 0; l < n; ++l) v[l] -= dot * V[i][l];
            }
        }
        
        double norm = vector_normalize(v, n);
        if (norm >= 1e-15) return;
        
        if (attempt < max_attempts - 1) {
            for (int l = 0; l < n; ++l) {
                v[l] = ((double)rand() / RAND_MAX) - 0.5;
            }
        }
    }
    vector_normalize(v, n);
}

// Check if matrix is symmetric
bool is_symmetric(double** T, int dim, double tol = 1e-10) {
    for (int i = 0; i < dim; i++) {
        for (int j = i + 1; j < dim; j++) {
            if (fabs(T[i][j] - T[j][i]) > tol) {
                return false;
            }
        }
    }
    return true;
}

// LAPACK-based eigensolver for small subspace problems
void small_eigensolver(double** T, int dim, int k, double* out_vals, double** out_vecs, int max_iter, double tol) {
    // Convert to column-major format for LAPACK
    double* A = (double*)malloc(dim * dim * sizeof(double));
    for (int i = 0; i < dim; i++) {
        for (int j = 0; j < dim; j++) {
            A[j * dim + i] = T[i][j];
        }
    }
    
    if (is_symmetric(T, dim)) {
        // Use dsyev for symmetric matrices (faster and more accurate)
        double* w = (double*)malloc(dim * sizeof(double));
        char jobz = 'V';  // Compute eigenvectors
        char uplo = 'U';  // Upper triangle
        int lwork = -1;
        double work_query;
        int info;
        
        // Workspace query
        dsyev_(&jobz, &uplo, &dim, A, &dim, w, &work_query, &lwork, &info);
        lwork = (int)work_query;
        double* work = (double*)malloc(lwork * sizeof(double));
        
        // Compute eigenvalues and eigenvectors
        dsyev_(&jobz, &uplo, &dim, A, &dim, w, work, &lwork, &info);
        
        if (info != 0) {
            fprintf(stderr, "LAPACK dsyev failed with info=%d\n", info);
            free(A); free(w); free(work);
            return;
        }
        
        // Extract k smallest (already sorted ascending)
        for (int i = 0; i < k; i++) {
            out_vals[i] = w[i];
            for (int j = 0; j < dim; j++) {
                out_vecs[i][j] = A[i * dim + j];
            }
        }
        
        free(w); free(work);
    } else {
        // Use dgeev for general matrices
        double* wr = (double*)malloc(dim * sizeof(double));
        double* wi = (double*)malloc(dim * sizeof(double));
        double* vl = (double*)malloc(dim * dim * sizeof(double));
        double* vr = (double*)malloc(dim * dim * sizeof(double));
        char jobvl = 'N';
        char jobvr = 'V';
        int lwork = -1;
        double work_query;
        int info;
        
        // Workspace query
        dgeev_(&jobvl, &jobvr, &dim, A, &dim, wr, wi, vl, &dim, vr, &dim, &work_query, &lwork, &info);
        lwork = (int)work_query;
        double* work = (double*)malloc(lwork * sizeof(double));
        
        // Compute eigenvalues and eigenvectors
        dgeev_(&jobvl, &jobvr, &dim, A, &dim, wr, wi, vl, &dim, vr, &dim, work, &lwork, &info);
        
        if (info != 0) {
            fprintf(stderr, "LAPACK dgeev failed with info=%d\n", info);
            free(A); free(wr); free(wi); free(vl); free(vr); free(work);
            return;
        }
        
        // Sort by real part
        int* idx = (int*)malloc(dim * sizeof(int));
        for (int i = 0; i < dim; i++) idx[i] = i;
        for (int i = 0; i < dim - 1; i++) {
            for (int j = i + 1; j < dim; j++) {
                if (wr[idx[j]] < wr[idx[i]]) {
                    int tmp = idx[i];
                    idx[i] = idx[j];
                    idx[j] = tmp;
                }
            }
        }
        
        // Extract k smallest
        for (int i = 0; i < k; i++) {
            int orig = idx[i];
            if (fabs(wi[orig]) > 1e-10) {
                fprintf(stderr, "Warning: Complex eigenvalue at index %d (imag=%.2e)\n", i, wi[orig]);
            }
            out_vals[i] = wr[orig];
            for (int j = 0; j < dim; j++) {
                out_vecs[i][j] = vr[orig * dim + j];
            }
        }
        
        free(wr); free(wi); free(vl); free(vr); free(work); free(idx);
    }
    
    free(A);
}

// Davidson algorithm with improved orthogonalization
EigenResult* davidson_algorithm(double** H, int n, int k, int max_iter, double tol) {
    // For small matrices, use direct LAPACK solver instead
    if (n <= 100 || n <= 3 * k) {
        EigenResult* result = (EigenResult*)malloc(sizeof(EigenResult));
        result->n_eigs = k;
        result->matrix_size = n;
        result->values = (double*)malloc(k * sizeof(double));
        result->vectors = (double**)malloc(k * sizeof(double*));
        for (int i = 0; i < k; i++) {
            result->vectors[i] = (double*)malloc(n * sizeof(double));
        }
        
        small_eigensolver(H, n, k, result->values, result->vectors, max_iter, tol);
        return result;
    }
    
    // Allocate result
    EigenResult* result = (EigenResult*)malloc(sizeof(EigenResult));
    result->n_eigs = k;
    result->matrix_size = n;
    result->values = (double*)malloc(k * sizeof(double));
    result->vectors = (double**)malloc(k * sizeof(double*));
    for (int i = 0; i < k; i++) {
        result->vectors[i] = (double*)malloc(n * sizeof(double));
    }

    // Diagonal preconditioner
    double* diag = (double*)malloc(n * sizeof(double));
    for (int i = 0; i < n; i++) diag[i] = H[i][i];

    // Subspace parameters
    int m_max = (n < 4 * k + 10) ? n : (4 * k + 10);
    int m_init = (n < 2 * k) ? n : (2 * k);
    if (m_init < k + 2) m_init = (n < k + 2) ? n : (k + 2);

    // Allocate subspace
    double** V = (double**)malloc(m_max * sizeof(double*));
    double** HV = (double**)malloc(m_max * sizeof(double*));
    for (int i = 0; i < m_max; ++i) {
        V[i] = (double*)malloc(n * sizeof(double));
        HV[i] = (double*)malloc(n * sizeof(double));
    }

    // Initialize with random vectors
    for (int i = 0; i < m_init; ++i) {
        for (int j = 0; j < n; ++j) {
            V[i][j] = ((double)rand() / RAND_MAX) - 0.5;
        }
    }
    orthonormalize(V, m_init, n);

    int current_dim = m_init;
    double* r = (double*)malloc(n * sizeof(double));
    double* tmp = (double*)malloc(n * sizeof(double));
    bool* converged = (bool*)malloc(k * sizeof(bool));
    double* residual_norms = (double*)malloc(k * sizeof(double));
    for (int i = 0; i < k; ++i) converged[i] = false;

    // Main Davidson loop
    for (int iter = 0; iter < max_iter; ++iter) {
        // Compute H*V for current subspace
        for (int j = 0; j < current_dim; ++j) {
            matrix_vector_mult(H, V[j], HV[j], n);
        }
        
        // Build subspace matrix T = V^T * H * V
        double** T = (double**)malloc(current_dim * sizeof(double*));
        for (int i = 0; i < current_dim; ++i) {
            T[i] = (double*)malloc(current_dim * sizeof(double));
            for (int j = 0; j < current_dim; ++j) {
                T[i][j] = vector_dot(V[i], HV[j], n);
            }
        }
        
        // Solve subspace eigenproblem with LAPACK
        double* theta = (double*)malloc(current_dim * sizeof(double));
        double** y = (double**)malloc(k * sizeof(double*));
        for (int i = 0; i < k; ++i) {
            y[i] = (double*)malloc(current_dim * sizeof(double));
        }
        small_eigensolver(T, current_dim, k, theta, y, 200, 1e-8);
        
        // Compute Ritz vectors and residuals
        double** new_vecs = (double**)malloc(k * sizeof(double*));
        for (int i = 0; i < k; ++i) {
            new_vecs[i] = (double*)malloc(n * sizeof(double));
        }
        int n_new = 0;

        for (int eig = 0; eig < k; ++eig) {
            // Ritz vector: u = V * y
            for (int i = 0; i < n; ++i) tmp[i] = 0.0;
            for (int j = 0; j < current_dim; ++j) {
                for (int i = 0; i < n; ++i) {
                    tmp[i] += y[eig][j] * V[j][i];
                }
            }
            
            // Orthogonalize against previously converged eigenvectors
            for (int prev = 0; prev < eig; ++prev) {
                double overlap = vector_dot(result->vectors[prev], tmp, n);
                for (int i = 0; i < n; ++i) {
                    tmp[i] -= overlap * result->vectors[prev][i];
                }
            }
            
            vector_normalize(tmp, n);

            // Residual: r = H*u - theta*u
            matrix_vector_mult(H, tmp, r, n);
            double res_norm = 0.0;
            for (int i = 0; i < n; ++i) {
                r[i] = r[i] - theta[eig] * tmp[i];
                res_norm += r[i] * r[i];
            }
            res_norm = sqrt(res_norm);
            residual_norms[eig] = res_norm;

            // Store Ritz pair
            for (int i = 0; i < n; ++i) {
                result->vectors[eig][i] = tmp[i];
            }
            result->values[eig] = theta[eig];

            // Track convergence per eigenvalue
            converged[eig] = (res_norm < tol);

            // Correction vector (only for non-converged eigenvalues)
            if (!converged[eig] && current_dim < m_max) {
                for (int i = 0; i < n; ++i) {
                    double denom = diag[i] - theta[eig];
                    if (fabs(denom) < 1e-12) {
                        new_vecs[n_new][i] = r[i];
                    } else {
                        new_vecs[n_new][i] = r[i] / denom;
                    }
                }
                n_new++;
            }
        }

        // Orthogonalize new vectors against each other first
        if (n_new > 0) {
            orthonormalize(new_vecs, n_new, n);
            
            // Then orthogonalize each against the subspace V
            int valid_new = 0;
            for (int a = 0; a < n_new; ++a) {
                orthonormalize_against(V, current_dim, new_vecs[a], n);
                
                // Check if vector became too small after orthogonalization
                double norm = vector_norm(new_vecs[a], n);
                if (norm >= 1e-10) {
                    if (valid_new != a) {
                        double* temp = new_vecs[valid_new];
                        new_vecs[valid_new] = new_vecs[a];
                        new_vecs[a] = temp;
                    }
                    valid_new++;
                }
            }
            n_new = valid_new;
        }

        int T_dim = current_dim;
        
        // Check if all eigenvalues converged
        bool all_converged = true;
        for (int i = 0; i < k; ++i) {
            if (!converged[i]) {
                all_converged = false;
                break;
            }
        }

        
        if (all_converged) {
            // Sort results
            for (int i = 0; i < k; ++i) {
                int min_idx = i;
                for (int j = i + 1; j < k; ++j) {
                    if (result->values[j] < result->values[min_idx]) {
                        min_idx = j;
                    }
                }
                if (min_idx != i) {
                    double tmpv = result->values[i];
                    result->values[i] = result->values[min_idx];
                    result->values[min_idx] = tmpv;
                    
                    double* tmpvec = result->vectors[i];
                    result->vectors[i] = result->vectors[min_idx];
                    result->vectors[min_idx] = tmpvec;
                }
            }

            for (int i = 0; i < k; ++i) free(new_vecs[i]);
            free(new_vecs);
            for (int i = 0; i < T_dim; ++i) free(T[i]);
            free(T);
            for (int i = 0; i < k; ++i) free(y[i]);
            free(y);
            free(theta);
            printf("Converged at iteration %d\n", iter);
            break;
        }
        
        // Add new vectors to subspace
        for (int a = 0; a < n_new && current_dim < m_max; ++a) {
            for (int i = 0; i < n; ++i) {
                V[current_dim][i] = new_vecs[a][i];
            }
            current_dim++;
        }

        if (current_dim >= m_max - 1) {
            int new_dim = m_init;
            
            // Keep k converged Ritz vectors
            for (int j = 0; j < k; ++j) {
                for (int i = 0; i < n; ++i) {
                    V[j][i] = result->vectors[j][i];
                }
            }
            
            // Add orthogonal random vectors
            for (int j = k; j < new_dim; ++j) {
                for (int i = 0; i < n; ++i) {
                    V[j][i] = ((double)rand() / RAND_MAX) - 0.5;
                }
                // Orthogonalize against converged vectors explicitly
                orthonormalize_against(V, j, V[j], n);
            }
            
            current_dim = new_dim;
        }
        
        // Cleanup iteration
        for (int i = 0; i < k; ++i) free(new_vecs[i]);
        free(new_vecs);
        for (int i = 0; i < T_dim; ++i) free(T[i]);
        free(T);
        for (int i = 0; i < k; ++i) free(y[i]);
        free(y);
        free(theta);
    }

    // Final sort
    for (int i = 0; i < k; ++i) {
        int min_idx = i;
        for (int j = i + 1; j < k; ++j) {
            if (result->values[j] < result->values[min_idx]) {
                min_idx = j;
            }
        }
        if (min_idx != i) {
            double tmpv = result->values[i];
            result->values[i] = result->values[min_idx];
            result->values[min_idx] = tmpv;
            
            double* tmpvec = result->vectors[i];
            result->vectors[i] = result->vectors[min_idx];
            result->vectors[min_idx] = tmpvec;
        }
    }

    // Cleanup
    for (int i = 0; i < m_max; ++i) {
        free(V[i]);
        free(HV[i]);
    }
    free(V);
    free(HV);
    free(diag);
    free(r);
    free(tmp);
    free(converged);
    free(residual_norms);

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