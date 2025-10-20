"""
Utility functions for 2x2 matrices.
Tested to work with float32 and float64.
"""

import numpy as np


EPS = 1e-12

def determinant_2x2_matrices(m):
    """
    Compute determinants of 2x2 matrices.
    
    Parameters
    ----------
    m : ndarray, shape (..., 2, 2)
        Input matrices with arbitrary leading dimensions.
        
    Returns
    -------
    ndarray, shape (...)
        Determinants with same leading dimensions as input.
    """
    original_shape = m.shape[:-2]
    m_flat = m.reshape(-1, 2, 2)
    det = m_flat[:, 0, 0] * m_flat[:, 1, 1] - m_flat[:, 0, 1] * m_flat[:, 1, 0]
    return det.reshape(original_shape)

def det_safe(det, EPS=EPS):
    sign = np.where(det >= 0, 1, -1)
    return np.where(np.abs(det) < EPS, EPS * sign, det)

def inverse_2x2_matrices(matrices, safe=False, EPS=EPS):
    """
    Compute inverses of 2x2 matrices.
    
    Parameters
    ----------
    matrices : ndarray, shape (..., 2, 2)
        Input matrices with arbitrary leading dimensions.
    safe : bool, optional
        If True, safeguard against near-zero determinants.
    EPS : float, optional
        Epsilon value for safe mode.
        
    Returns
    -------
    ndarray, shape (..., 2, 2)
        Inverse matrices with same shape as input.
    """
    original_shape = matrices.shape
    matrices_flat = matrices.reshape(-1, 2, 2)
    
    det = determinant_2x2_matrices(matrices_flat)

    if safe:
        det = det_safe(det, EPS=EPS)

    inv_matrices = np.empty_like(matrices_flat)
    inv_matrices[:, 0, 0] = matrices_flat[:, 1, 1] / det
    inv_matrices[:, 1, 1] = matrices_flat[:, 0, 0] / det
    inv_matrices[:, 0, 1] = -matrices_flat[:, 0, 1] / det
    inv_matrices[:, 1, 0] = -matrices_flat[:, 1, 0] / det

    return inv_matrices.reshape(original_shape)

def mahalanobis_2x2_matrices(matrices, diffs, is_inverse=True, squared=True):
    """
    Compute Mahalanobis distances for multiple 2D vectors and 2x2 matrices.
    
    Parameters
    ----------
    matrices : ndarray, shape (..., 2, 2)
        Covariance or inverse covariance matrices with arbitrary leading dimensions.
    diffs : ndarray, shape (..., 2)
        2D vectors with same leading dimensions as matrices.
    is_inverse : bool, optional
        If True, matrices are treated as inverse covariances. If False, they are treated as covariances.

    Returns
    -------
    ndarray, shape (...)
        Mahalanobis distances.
    """
    if is_inverse:
        inv_matrices = matrices
    else:
        inv_matrices = inverse_2x2_matrices(matrices)

    # Reshape leading dims and store original shape
    original_shape = diffs.shape[:-1]
    diffs_flat = diffs.reshape(-1, 2)
    inv_matrices_flat = inv_matrices.reshape(-1, 2, 2)

    # Compute Mahalanobis distances using quadratic 2x2 form
    Av_0 = inv_matrices_flat[:, 0, 0] * diffs_flat[:, 0] + inv_matrices_flat[:, 0, 1] * diffs_flat[:, 1]
    Av_1 = inv_matrices_flat[:, 1, 0] * diffs_flat[:, 0] + inv_matrices_flat[:, 1, 1] * diffs_flat[:, 1]

    mahalanobis_sq = diffs_flat[:, 0] * Av_0 + diffs_flat[:, 1] * Av_1
    if not squared:
        mahalanobis_sq[mahalanobis_sq < 0] = 0.0
        mahalanobis_sq = np.sqrt(mahalanobis_sq)
    return mahalanobis_sq.reshape(original_shape)

def eigh_2x2_matrices(matrices, EPS=1e-12):
    """
    Vectorized eigendecomposition for 2x2 symmetric matrices.
    
    Parameters
    ----------
    matrices : ndarray, shape (..., 2, 2)
        Symmetric matrices with arbitrary leading dimensions.
    EPS : float, optional
        Epsilon for numerical stability.
        
    Returns
    -------
    w : ndarray, shape (..., 2)
        Eigenvalues (smaller first).
    v : ndarray, shape (..., 2, 2)
        Eigenvectors as columns.
    """
    original_shape = matrices.shape[:-2]
    matrices_flat = matrices.reshape(-1, 2, 2)
    n = len(matrices_flat)

    # Check symmetry:
    if not np.allclose(matrices_flat[:, 0, 1], matrices_flat[:, 1, 0]):
        raise ValueError("Input matrices must be symmetric.")
    
    a = matrices_flat[:, 0, 0]
    b = matrices_flat[:, 0, 1]  # Assume symmetric: b == c
    d = matrices_flat[:, 1, 1]
    
    trace = a + d
    det = a * d - b * b
    discriminant = np.maximum(trace**2 - 4 * det, 0)
    sqrt_disc = np.sqrt(discriminant)
    
    # Eigenvalues
    w = np.empty((n, 2), dtype=matrices_flat.dtype)
    w[:, 0] = (trace - sqrt_disc) / 2  # Smaller first
    w[:, 1] = (trace + sqrt_disc) / 2  # Larger second
    
    # Eigenvectors (vectorized)
    v = np.zeros((n, 2, 2), dtype=matrices_flat.dtype)
    
    # Use b as denominator when non-zero
    use_b = np.abs(b) > EPS
    
    # v1 for λ1
    v[use_b, 0, 0] = b[use_b]
    v[use_b, 1, 0] = w[use_b, 0] - a[use_b]
    v[~use_b, 0, 0] = 1.0  # Diagonal case
    
    # v2 for λ2
    v[use_b, 0, 1] = b[use_b]
    v[use_b, 1, 1] = w[use_b, 1] - a[use_b]
    v[~use_b, 1, 1] = 1.0  # Diagonal case
    
    # Normalize
    norms = np.sqrt(np.sum(v**2, axis=1, keepdims=True))
    v /= norms + 1e-20  # Avoid division by zero

    # Apply numpy convention
    for i in range(2):
        first_elem = v[:, 0, i]
        second_elem = v[:, 1, i]
        
        # Case 1: First element is significantly non-zero and negative
        flip_first = (np.abs(first_elem) > EPS) & (first_elem < 0)
        v[flip_first, :, i] *= -1
        
        # Case 2: First element near zero, make second element positive
        first_near_zero = np.abs(first_elem) <= EPS
        flip_second = first_near_zero & (second_elem < 0)
        v[flip_second, :, i] *= -1
    
    return w.reshape(original_shape + (2,)), v.reshape(original_shape + (2, 2))

def solve_2x2_matrices(A, b, safe=False, EPS=1e-12):
    """
    Solve Ax = b for multiple 2x2 matrices A and vectors b.
    
    Parameters
    ----------
    A : ndarray, shape (..., 2, 2)
        Coefficient matrices with arbitrary leading dimensions.
    b : ndarray, shape (..., 2)
        Right-hand side vectors.
    safe : bool, optional
        If True, safeguard against near-zero determinants.
    EPS : float, optional
        Epsilon value for safe mode.
        
    Returns
    -------
    ndarray, shape (..., 2)
        Solution vectors.
    """
    original_shape = A.shape[:-2]
    A_flat = A.reshape(-1, 2, 2)
    b_flat = b.reshape(-1, 2)
    
    det = determinant_2x2_matrices(A_flat)
    if safe:
        det = det_safe(det, EPS=EPS)

    x = np.empty_like(b_flat)

    x[:, 0] = (A_flat[:, 1, 1] * b_flat[:, 0] - A_flat[:, 0, 1] * b_flat[:, 1]) / det
    x[:, 1] = (A_flat[:, 0, 0] * b_flat[:, 1] - A_flat[:, 1, 0] * b_flat[:, 0]) / det
    return x.reshape(original_shape + (2,))

def norm_2d(vectors, axis=-1):
    """
    Compute Euclidean norm for 2D vectors.
    vectors: shape (..., 2), axis: axis along which to compute norm.
    """
    return np.sqrt(np.sum(vectors**2, axis=axis))


if __name__ == "__main__":
    # Benchmarking numpy vs 2x2 functions

    import time
    
    N = 10_000_000
    array_dtype = np.float32
    ms_precision = 2  # Number of decimal places for ms output

    # print settings (N, dtype) and a nice header
    sep = "-" * 20
    print(f'{sep} Start of 2x2 matrix operations benchmark {sep}')
    print(f"Number of matrices: {N}")
    print(f"Data type: {array_dtype}")

    if array_dtype == np.float32:
        rtol, atol = 1e-4, 1e-5
    elif array_dtype == np.float64:
        rtol, atol = 1e-5, 1e-8

    def fmt_ms(seconds):
        return f"{seconds * 1000:.{ms_precision}f} ms"

    # Determinant
    print("\n" + "=" * 70)
    print("Determinant")
    print("=" * 70)
    A = np.random.rand(N, 2, 2).astype(array_dtype)

    t0 = time.perf_counter()
    det1 = determinant_2x2_matrices(A)
    t1 = time.perf_counter()
    det2 = np.linalg.det(A)
    t2 = time.perf_counter()

    # Dtype check
    if det1.dtype != array_dtype:
        print(f"[WARNING] Custom determinant dtype {det1.dtype} != {array_dtype}")
    if det2.dtype != array_dtype:
        print(f"[WARNING] Numpy determinant dtype {det2.dtype} != {array_dtype}")

    if not np.allclose(det1, det2, rtol=rtol, atol=atol):
        print("Determinants do not match!")
        abs_diff = np.abs(det1 - det2)
        max_diff = np.max(abs_diff)
        idx = np.argmax(abs_diff)
        mismatch_mask = ~np.isclose(det1, det2, rtol=rtol, atol=atol)
        mismatch_count = np.sum(mismatch_mask)
        print(f"  Max abs diff: {max_diff}")
        print(f"  Custom value: {det1[idx]}, Numpy value: {det2[idx]}")
        print(f"  Mismatch count: {mismatch_count}")

    print(f"Custom: {fmt_ms(t1 - t0)}")
    print(f"Numpy:  {fmt_ms(t2 - t1)}")
    print(f"Speedup: {(t2 - t1) / (t1 - t0):.{ms_precision}f}x")

    # Inverse
    print("\n" + "=" * 70)
    print("Inverse")
    print("=" * 70)
    non_invertible = np.abs(determinant_2x2_matrices(A)) < 1e-3
    A[non_invertible] += np.eye(2) * 1e-2

    t0 = time.perf_counter()
    inv1 = inverse_2x2_matrices(A)
    t1 = time.perf_counter()
    inv2 = np.linalg.inv(A)
    t2 = time.perf_counter()

    # Dtype check
    if inv1.dtype != array_dtype:
        print(f"[WARNING] Custom inverse dtype {inv1.dtype} != {array_dtype}")
    if inv2.dtype != array_dtype:
        print(f"[WARNING] Numpy inverse dtype {inv2.dtype} != {array_dtype}")

    if not np.allclose(inv1, inv2, rtol=rtol, atol=atol):
        print("Inverses do not match!")
        abs_diff = np.abs(inv1 - inv2)
        max_diff = np.max(abs_diff)
        idx = np.unravel_index(np.argmax(abs_diff), inv1.shape)
        mismatch_mask = ~np.isclose(inv1, inv2, rtol=rtol, atol=atol)
        mismatch_count = np.sum(mismatch_mask)
        print(f"  Max abs diff: {max_diff}")
        print(f"  Custom value: {inv1[idx]}, Numpy value: {inv2[idx]}")
        print(f"  Mismatch count: {mismatch_count}")

    print(f"Custom: {fmt_ms(t1 - t0)}")
    print(f"Numpy:  {fmt_ms(t2 - t1)}")
    print(f"Speedup: {(t2 - t1) / (t1 - t0):.{ms_precision}f}x")

    # Eigendecomposition
    print("\n" + "=" * 70)
    print("Eigendecomposition")
    print("=" * 70)
    A = A + np.transpose(A, (0, 2, 1))

    t0 = time.perf_counter()
    w1, v1 = eigh_2x2_matrices(A)
    t1 = time.perf_counter()
    w2, v2 = np.linalg.eigh(A)
    t2 = time.perf_counter()

    # Dtype check
    if w1.dtype != array_dtype:
        print(f"[WARNING] Custom eigenvalues dtype {w1.dtype} != {array_dtype}")
    if w2.dtype != array_dtype:
        print(f"[WARNING] Numpy eigenvalues dtype {w2.dtype} != {array_dtype}")
    if v1.dtype != array_dtype:
        print(f"[WARNING] Custom eigenvectors dtype {v1.dtype} != {array_dtype}")
    if v2.dtype != array_dtype:
        print(f"[WARNING] Numpy eigenvectors dtype {v2.dtype} != {array_dtype}")

    if not np.allclose(w1, w2, rtol=rtol, atol=atol):
        print("Eigenvalues do not match!")
        abs_diff = np.abs(w1 - w2)
        max_diff = np.max(abs_diff)
        idx = np.unravel_index(np.argmax(abs_diff), w1.shape)
        mismatch_mask = ~np.isclose(w1, w2, rtol=rtol, atol=atol)
        mismatch_count = np.sum(mismatch_mask)
        print(f"  Max abs diff: {max_diff}")
        print(f"  Custom value: {w1[idx]}, Numpy value: {w2[idx]}")
        print(f"  Mismatch count: {mismatch_count}")

    if not np.allclose(v1, v2, rtol=rtol, atol=atol):
        print("Eigenvectors do not match!")
        abs_diff = np.abs(v1 - v2)
        max_diff = np.max(abs_diff)
        idx = np.unravel_index(np.argmax(abs_diff), v1.shape)
        mismatch_mask = ~np.isclose(v1, v2, rtol=rtol, atol=atol)
        mismatch_count = np.sum(mismatch_mask)
        print(f"  Max abs diff: {max_diff}")
        print(f"  Custom value: {v1[idx]}, Numpy value: {v2[idx]}")
        print(f"  Mismatch count: {mismatch_count}")

    print(f"Custom: {fmt_ms(t1 - t0)}")
    print(f"Numpy:  {fmt_ms(t2 - t1)}")
    print(f"Speedup: {(t2 - t1) / (t1 - t0):.{ms_precision}f}x")

    # Solve Ax = b
    print("\n" + "=" * 70)
    print("Solve Ax = b")
    print("=" * 70)
    A = np.random.rand(N, 2, 2).astype(array_dtype)
    b = np.random.rand(N, 2).astype(array_dtype)

    # Make invertible
    non_invertible = np.abs(determinant_2x2_matrices(A)) < 1e-3
    A[non_invertible] += np.eye(2, dtype=array_dtype) * 1e-2

    t0 = time.perf_counter()
    x1 = solve_2x2_matrices(A, b)
    t1 = time.perf_counter()
    x2 = np.linalg.solve(A, b[:,:, None])[:, :, 0]
    t2 = time.perf_counter()

    # Dtype check
    if x1.dtype != array_dtype:
        print(f"[WARNING] Custom solve dtype {x1.dtype} != {array_dtype}")
    if x2.dtype != array_dtype:
        print(f"[WARNING] Numpy solve dtype {x2.dtype} != {array_dtype}")

    if not np.allclose(x1, x2, rtol=rtol, atol=atol):
        print("Solutions do not match!")
        abs_diff = np.abs(x1 - x2)
        max_diff = np.max(abs_diff)
        idx = np.unravel_index(np.argmax(abs_diff), x1.shape)
        mismatch_mask = ~np.isclose(x1, x2, rtol=rtol, atol=atol)
        mismatch_count = np.sum(mismatch_mask)
        print(f"  Max abs diff: {max_diff}")
        print(f"  Custom value: {x1[idx]}, Numpy value: {x2[idx]}")
        print(f"  Mismatch count: {mismatch_count}")

    print(f"Custom: {fmt_ms(t1 - t0)}")
    print(f"Numpy:  {fmt_ms(t2 - t1)}")
    print(f"Speedup: {(t2 - t1) / (t1 - t0):.{ms_precision}f}x")

    # Norm 2D
    print("\n" + "=" * 70)
    print("Norm 2D")
    print("=" * 70)
    t0 = time.perf_counter()
    n1 = norm_2d(b)
    t1 = time.perf_counter()
    n2 = np.linalg.norm(b, axis=1)
    t2 = time.perf_counter()

    # Dtype check
    if n1.dtype != array_dtype:
        print(f"[WARNING] Custom norm dtype {n1.dtype} != {array_dtype}")
    if n2.dtype != array_dtype:
        print(f"[WARNING] Numpy norm dtype {n2.dtype} != {array_dtype}")

    if not np.allclose(n1, n2, rtol=rtol, atol=atol):
        print("Norms do not match!")
        abs_diff = np.abs(n1 - n2)
        max_diff = np.max(abs_diff)
        idx = np.argmax(abs_diff)
        mismatch_mask = ~np.isclose(n1, n2, rtol=rtol, atol=atol)
        mismatch_count = np.sum(mismatch_mask)
        print(f"  Max abs diff: {max_diff}")
        print(f"  Custom value: {n1[idx]}, Numpy value: {n2[idx]}")
        print(f"  Mismatch count: {mismatch_count}")

    print(f"Custom: {fmt_ms(t1 - t0)}")
    print(f"Numpy:  {fmt_ms(t2 - t1)}")
    print(f"Speedup: {(t2 - t1) / (t1 - t0):.{ms_precision}f}x")

    # Test with arbitrary leading dimensions
    print("\n" + "=" * 70)
    print("Testing with arbitrary leading dimensions")
    print("=" * 70)
    
    # Test with shape (3, 4, 2, 2) - simulating a 3x4 grid of 2x2 matrices
    test_matrices = np.random.rand(3, 4, 2, 2).astype(np.float32)
    test_matrices = test_matrices + np.transpose(test_matrices, (0, 1, 3, 2))  # Make symmetric
    
    print(f"Test matrices shape: {test_matrices.shape}")
    
    # Test determinant
    det = determinant_2x2_matrices(test_matrices)
    print(f"Determinant output shape: {det.shape} (expected: (3, 4))")
    assert det.shape == (3, 4), f"Expected shape (3, 4), got {det.shape}"
    
    # Test inverse
    inv = inverse_2x2_matrices(test_matrices)
    print(f"Inverse output shape: {inv.shape} (expected: (3, 4, 2, 2))")
    assert inv.shape == (3, 4, 2, 2), f"Expected shape (3, 4, 2, 2), got {inv.shape}"
    
    # Test eigh
    w, v = eigh_2x2_matrices(test_matrices)
    print(f"Eigenvalues shape: {w.shape} (expected: (3, 4, 2))")
    print(f"Eigenvectors shape: {v.shape} (expected: (3, 4, 2, 2))")
    assert w.shape == (3, 4, 2), f"Expected shape (3, 4, 2), got {w.shape}"
    assert v.shape == (3, 4, 2, 2), f"Expected shape (3, 4, 2, 2), got {v.shape}"
    
    # Test solve
    test_b = np.random.rand(3, 4, 2).astype(np.float32)
    x = solve_2x2_matrices(test_matrices, test_b)
    print(f"Solve output shape: {x.shape} (expected: (3, 4, 2))")
    assert x.shape == (3, 4, 2), f"Expected shape (3, 4, 2), got {x.shape}"
    
    # Test with single matrix (0D leading dimensions)
    single_matrix = np.random.rand(2, 2).astype(np.float32)
    single_matrix = single_matrix + single_matrix.T
    det_single = determinant_2x2_matrices(single_matrix)
    print(f"Single matrix determinant shape: {det_single.shape} (expected: ())")
    assert det_single.shape == (), f"Expected shape (), got {det_single.shape}"
    
    print("\n✓ All shape tests passed!")


