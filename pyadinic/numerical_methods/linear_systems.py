import numpy as np

def gaussian_elimination(A, b, tolerance=1e-12):

    # Validar A
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("A must contain only numeric values")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Validar b
    try:
        b = np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("b must contain only numeric values")

    if b.ndim != 1:
        raise ValueError("b must be a 1D array")

    if len(b) != A.shape[0]:
        raise ValueError(
            "b must have the same number of elements as A has rows"
        )

    # Validar tolerance
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")

    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")

    # Copias para no modificar los argumentos originales
    A = A.copy()
    b = b.copy()

    n = len(b)

    # Eliminación hacia adelante
    for k in range(n - 1):

        # Pivoteo parcial
        pivot_row = k + np.argmax(np.abs(A[k:, k]))

        if abs(A[pivot_row, k]) < tolerance:
            raise ValueError(
                "Matrix is singular or nearly singular"
            )

        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        # Eliminación
        for i in range(k + 1, n):

            factor = A[i, k] / A[k, k]

            A[i, k:] -= factor * A[k, k:]
            b[i] -= factor * b[k]

    # Verificar último pivote
    if abs(A[-1, -1]) < tolerance:
        raise ValueError(
            "Matrix is singular or nearly singular"
        )

    # Sustitución hacia atrás
    x = np.zeros(n)

    for i in range(n - 1, -1, -1):

        x[i] = (
            b[i] - np.dot(A[i, i + 1:], x[i + 1:])
        ) / A[i, i]

    return x

def gauss_jordan(A, b, tolerance=1e-12):
    # Validar A
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("A must contain only numeric values")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Validar b
    try:
        b = np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("b must contain only numeric values")

    if b.ndim != 1:
        raise ValueError("b must be a 1D array")

    if len(b) != A.shape[0]:
        raise ValueError(
            "b must have the same number of elements as A has rows"
        )

    # Validar tolerance
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")

    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")

    # Copias para no modificar los argumentos originales
    A = A.copy()
    b = b.copy()

    n = len(b)

    # Eliminación Gauss-Jordan
    for k in range(n):

        # Pivoteo parcial
        pivot_row = k + np.argmax(np.abs(A[k:, k]))

        if abs(A[pivot_row, k]) < tolerance:
            raise ValueError(
                "Matrix is singular or nearly singular"
            )

        if pivot_row != k:
            A[[k, pivot_row]] = A[[pivot_row, k]]
            b[[k, pivot_row]] = b[[pivot_row, k]]

        pivot = A[k, k]

        A[k, :] /= pivot
        b[k] /= pivot

        for i in range(n):

            if i == k:
                continue

            factor = A[i, k]

            A[i, :] -= factor * A[k, :]
            b[i] -= factor * b[k]

    return b


def lu_decomposition(A, tolerance=1e-12, pivoting=False):
    # Validar A
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("A must contain only numeric values")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Validar tolerance
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")

    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")

    # Validar pivoting
    if not isinstance(pivoting, bool):
        raise TypeError("pivoting must be a boolean")

    n = A.shape[0]
    A = A.copy()

    if pivoting:
        P = np.eye(n, dtype=float)
        L = np.eye(n, dtype=float)
        U = A.copy()

        for k in range(n):
            pivot_row = k + np.argmax(np.abs(U[k:, k]))

            if abs(U[pivot_row, k]) < tolerance:
                raise ValueError("Matrix is singular or nearly singular")

            if pivot_row != k:
                U[[k, pivot_row]] = U[[pivot_row, k]]
                P[[k, pivot_row]] = P[[pivot_row, k]]
                if k > 0:
                    L[[k, pivot_row], :k] = L[[pivot_row, k], :k]

            for i in range(k + 1, n):
                factor = U[i, k] / U[k, k]
                L[i, k] = factor
                U[i, k:] -= factor * U[k, k:]
                U[i, k] = 0.0

        if abs(U[-1, -1]) < tolerance:
            raise ValueError("Matrix is singular or nearly singular")

        return P, L, U

    else:
        L = np.eye(n, dtype=float)
        U = np.zeros((n, n), dtype=float)

        for i in range(n):
            for j in range(i, n):
                U[i, j] = A[i, j] - np.dot(L[i, :i], U[:i, j])

            if abs(U[i, i]) < tolerance:
                raise ValueError("Matrix is singular or nearly singular")

            for j in range(i + 1, n):
                L[j, i] = (A[j, i] - np.dot(L[j, :i], U[:i, i])) / U[i, i]

        return L, U


def lu_solve(A, b, tolerance=1e-12):
    # Validar A
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("A must contain only numeric values")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Validar b
    try:
        b = np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("b must contain only numeric values")

    if b.ndim != 1:
        raise ValueError("b must be a 1D array")

    if len(b) != A.shape[0]:
        raise ValueError(
            "b must have the same number of elements as A has rows"
        )

    # Validar tolerance
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")

    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")

    # Copias para no modificar los argumentos originales
    A = A.copy()
    b = b.copy()

    n = len(b)

    # Descomposición PA = LU con pivoteo parcial
    P, L, U = lu_decomposition(A, tolerance=tolerance, pivoting=True)

    # Permutación de b
    Pb = np.dot(P, b)

    # Sustitución hacia adelante: L y = Pb
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = Pb[i] - np.dot(L[i, :i], y[:i])

    # Sustitución hacia atrás: U x = y
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(U[i, i + 1:], x[i + 1:])) / U[i, i]

    return x


def cholesky_decomposition(A, tolerance=1e-12, lower=True):
    # Validar A
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("A must contain only numeric values")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Validar tolerance
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")

    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")

    if not isinstance(lower, bool):
        raise TypeError("lower must be a boolean")

    # Validar simetría
    if not np.allclose(A, A.T, atol=tolerance):
        raise ValueError("Matrix must be symmetric")

    n = A.shape[0]
    A = A.copy()
    L = np.zeros((n, n), dtype=float)

    for i in range(n):
        for j in range(i + 1):
            sum_k = np.dot(L[i, :j], L[j, :j])
            if i == j:
                val = A[i, i] - sum_k
                if val <= tolerance:
                    raise ValueError("Matrix is not positive definite")
                L[i, i] = np.sqrt(val)
            else:
                L[i, j] = (A[i, j] - sum_k) / L[j, j]

    if lower:
        return L
    else:
        return L.T


def cholesky_solve(A, b, tolerance=1e-12):
    # Validar A
    try:
        A = np.asarray(A, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("A must contain only numeric values")

    if A.ndim != 2:
        raise ValueError("A must be a 2D array")

    if A.shape[0] != A.shape[1]:
        raise ValueError("A must be a square matrix")

    # Validar b
    try:
        b = np.asarray(b, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("b must contain only numeric values")

    if b.ndim != 1:
        raise ValueError("b must be a 1D array")

    if len(b) != A.shape[0]:
        raise ValueError(
            "b must have the same number of elements as A has rows"
        )

    # Validar tolerance
    if not isinstance(tolerance, (int, float)) or isinstance(tolerance, bool):
        raise TypeError("tolerance must be numeric")

    if tolerance <= 0:
        raise ValueError("tolerance must be greater than zero")

    # Copias para no modificar los argumentos originales
    A = A.copy()
    b = b.copy()

    n = len(b)

    # Descomposición de Cholesky A = L @ L.T
    L = cholesky_decomposition(A, tolerance=tolerance, lower=True)

    # Sustitución hacia adelante: L y = b
    y = np.zeros(n, dtype=float)
    for i in range(n):
        y[i] = (b[i] - np.dot(L[i, :i], y[:i])) / L[i, i]

    # Sustitución hacia atrás: L.T x = y
    x = np.zeros(n, dtype=float)
    for i in range(n - 1, -1, -1):
        x[i] = (y[i] - np.dot(L[i + 1:, i], x[i + 1:])) / L[i, i]

    return x


def lu(A, b=None, tolerance=1e-12, pivoting=False):
    """
    Método LU para resolver Ax = b o calcular la descomposición LU de A.
    """
    if b is not None:
        return lu_solve(A, b, tolerance=tolerance)
    return lu_decomposition(A, tolerance=tolerance, pivoting=pivoting)


def cholesky(A, b=None, tolerance=1e-12, lower=True):
    """
    Método de Cholesky para resolver Ax = b o calcular la descomposición de Cholesky de A.
    """
    if b is not None:
        return cholesky_solve(A, b, tolerance=tolerance)
    return cholesky_decomposition(A, tolerance=tolerance, lower=lower)


lu_factorization = lu_decomposition
cholesky_factorization = cholesky_decomposition