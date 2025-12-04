import numpy as np
import time

# Use perf_counter for high precision timing
get_time = time.perf_counter

print("=== EIGENVALUE SOLVER TESTS (NumPy Baseline) ===\n")

# Test matrices - EXACT same as C++ test file
matrices = {
    "2x2": np.array([
        [3.0, 1.0],
        [0.0, 2.0]
    ]),
    
    "3x3": np.array([
        [2.0, 0.0, 1.0],
        [-1.0, 4.0, -1.0],
        [-1.0, 2.0, 0.0]
    ]),
    
    "4x4": np.array([
        [4.0, -1.0, 0.0, 0.0],
        [-1.0, 4.0, -1.0, 0.0],
        [0.0, -1.0, 4.0, -1.0],
        [0.0, 0.0, -1.0, 4.0]
    ]),
    
    "5x5": np.array([
        [5.0, 1.0, 0.0, 0.0, 0.0],
        [1.0, 4.0, 1.0, 0.0, 0.0],
        [0.0, 1.0, 3.0, 1.0, 0.0],
        [0.0, 0.0, 1.0, 2.0, 1.0],
        [0.0, 0.0, 0.0, 1.0, 1.0]
    ])
}

# SMALL MATRIX TESTS
print("SMALL MATRIX TESTS")
print("==================\n")

for name, A in matrices.items():
    print("=" * 70)
    print(f"{name} Matrix Test")
    print("=" * 70)
    
    print(f"{name} Matrix:")
    for row in A:
        print("  " + " ".join(f"{val:7.4f}" for val in row))
    print()
    
    start_time = get_time()
    
    # Compute ALL eigenvalues and eigenvectors (like LAPACK does)
    eigenvalues, eigenvectors = np.linalg.eig(A)
    
    # Sort by eigenvalue (ascending)
    idx = eigenvalues.argsort()
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]
    
    elapsed_time = get_time() - start_time
    
    print("Eigenvalues:")
    for i, val in enumerate(eigenvalues):
        print(f"  lambda_{i} = {val:.10f}")
    print()
    
    print("Eigenvectors (first few rows):")
    n_rows = min(4, A.shape[0])
    for i in range(n_rows):
        row_str = "  [" + ", ".join(f"{eigenvectors[i, j]:9.6f}" for j in range(A.shape[1])) + "]"
        print(row_str)
    print()
    
    print("Verification (||A*v - lambda*v||):")
    for i in range(len(eigenvalues)):
        v = eigenvectors[:, i]
        Av = A @ v
        lv = eigenvalues[i] * v
        residual = np.linalg.norm(Av - lv)
        status = "OK" if residual < 1e-8 else "FAIL"
        print(f"  lambda_{i}: ||A*v - lambda*v|| = {residual:.2e} {status}")
    print()
    
    print(f"Computation time: {elapsed_time*1000:.6f} ms\n")


# LARGE SPARSE MATRIX TESTS
print("\n" + "=" * 70)
print("LARGE SPARSE HAMILTONIAN TESTS")
print("=" * 70)
print("\nNOTE: NumPy computes ALL eigenvalues, not just k smallest.")
print("      This is much slower than Davidson for large sparse matrices!\n")

def create_sparse_hamiltonian(n, sparsity, seed):
    """Create same sparse Hamiltonian as C++ version"""
    np.random.seed(seed)
    H = np.zeros((n, n))
    
    # Diagonal: on-site energies (matches C++ exactly)
    for i in range(n):
        H[i, i] = 1.0 + np.random.rand() * 4.0
    
    # Off-diagonal: nearest neighbor hopping
    for i in range(n-1):
        val = -0.1 - np.random.rand() * 0.9
        H[i, i+1] = val
        H[i+1, i] = val
    
    # Add sparse random interactions
    n_random = int(n * n * sparsity)
    for _ in range(n_random):
        i = np.random.randint(0, n)
        j = np.random.randint(0, n)
        if i != j:
            val = (np.random.rand() - 0.5) * 1.0
            H[i, j] = val
            H[j, i] = val
    
    return H

# Test 1: 1000x1000
print("=" * 70)
print("1000x1000 Sparse Hamiltonian (k=5) Matrix Test")
print("=" * 70)

n = 1000
k = 5
print(f"Matrix size: {n}x{n} (sparse)")
print(f"Finding {k} smallest eigenvalues...")
print("WARNING: NumPy will compute ALL 1000 eigenvalues!\n")

H = create_sparse_hamiltonian(n, 0.001, 42)

start_time = get_time()
eigenvalues, eigenvectors = np.linalg.eig(H)
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx][:k]  # Take only k smallest
eigenvectors = eigenvectors[:, idx][:, :k]
elapsed_time = get_time() - start_time

print("Eigenvalues:")
for i in range(k):
    print(f"  lambda_{i} = {eigenvalues[i]:.10f}")
print()

print("Verification (||A*v - lambda*v||):")
for i in range(k):
    v = eigenvectors[:, i]
    Av = H @ v
    lv = eigenvalues[i] * v
    residual = np.linalg.norm(Av - lv)
    status = "OK" if residual < 1e-8 else "FAIL"
    print(f"  lambda_{i}: ||A*v - lambda*v|| = {residual:.2e} {status}")
print()

print(f"Computation time: {elapsed_time*1000:.6f} ms ({elapsed_time:.6f} seconds)\n")


# Test 2: 10000x10000
print("=" * 70)
print("10000x10000 Sparse Hamiltonian (k=5) Matrix Test")
print("=" * 70)

n = 10000
k = 5
print(f"Matrix size: {n}x{n} (sparse)")
print(f"Finding {k} smallest eigenvalues...")
print("WARNING: NumPy will compute ALL 10000 eigenvalues!")
print("         This will take a VERY long time...\n")

H = create_sparse_hamiltonian(n, 0.0001, 42)

start_time = get_time()
eigenvalues, eigenvectors = np.linalg.eig(H)
idx = eigenvalues.argsort()
eigenvalues = eigenvalues[idx][:k]
eigenvectors = eigenvectors[:, idx][:, :k]
elapsed_time = get_time() - start_time

print("Eigenvalues:")
for i in range(k):
    print(f"  lambda_{i} = {eigenvalues[i]:.10f}")
print()

print("Verification (||A*v - lambda*v||):")
for i in range(k):
    v = eigenvectors[:, i]
    Av = H @ v
    lv = eigenvalues[i] * v
    residual = np.linalg.norm(Av - lv)
    status = "OK" if residual < 1e-8 else "FAIL"
    print(f"  lambda_{i}: ||A*v - lambda*v|| = {residual:.2e} {status}")
print()

print(f"Computation time: {elapsed_time*1000:.6f} ms ({elapsed_time:.6f} seconds)\n")