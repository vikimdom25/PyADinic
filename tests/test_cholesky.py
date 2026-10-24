import numpy as np
import pytest

from pyadinic.numerical_methods.linear_systems import (
    cholesky_decomposition,
    cholesky_solve,
    cholesky,
    cholesky_factorization,
)


# ==============================================================================
# Tests for cholesky_solve
# ==============================================================================

def test_cholesky_solve_2x2():
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    b = np.array([
        8.0,
        7.0
    ])

    x = cholesky_solve(A, b)

    expected = np.array([
        1.25,
        1.5
    ])

    assert x == pytest.approx(expected)


def test_cholesky_solve_3x3():
    A = np.array([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0]
    ])

    b = np.array([
        1.0,
        2.0,
        3.0
    ])

    x = cholesky_solve(A, b)

    assert np.dot(A, x) == pytest.approx(b)


def test_cholesky_solve_accepts_lists():
    A = [
        [4.0, 2.0],
        [2.0, 3.0]
    ]

    b = [
        8.0,
        7.0
    ]

    x = cholesky_solve(A, b)

    expected = np.array([
        1.25,
        1.5
    ])

    assert x == pytest.approx(expected)


def test_cholesky_solve_does_not_modify_inputs():
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    b = np.array([
        8.0,
        7.0
    ])

    A_original = A.copy()
    b_original = b.copy()

    cholesky_solve(A, b)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)


def test_cholesky_solve_non_symmetric_matrix():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0]
    ])

    b = np.array([
        8.0,
        7.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_not_positive_definite():
    # Matriz simétrica pero indefinida (determinante < 0)
    A = np.array([
        [1.0, 2.0],
        [2.0, 1.0]
    ])

    b = np.array([
        3.0,
        3.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_singular_matrix():
    # Matriz simétrica semidefinida positiva (singular)
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])

    b = np.array([
        3.0,
        6.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_nearly_singular_matrix():
    A = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-14]
    ])

    b = np.array([
        2.0,
        2.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_invalid_A_dimension():
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_invalid_A_not_2d():
    A = np.array([
        1.0,
        2.0,
        3.0
    ])

    b = np.array([
        1.0,
        2.0,
        3.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_invalid_b_dimension():
    A = np.eye(2)

    b = np.array([
        [1.0],
        [2.0]
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_incompatible_dimensions():
    A = np.eye(3)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        cholesky_solve(A, b)


def test_cholesky_solve_invalid_A_values():
    A = np.array([
        [1.0, 2.0],
        [2.0, "text"]
    ], dtype=object)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(TypeError):
        cholesky_solve(A, b)


def test_cholesky_solve_invalid_b_values():
    A = np.eye(2)

    b = np.array([
        1.0,
        "text"
    ], dtype=object)

    with pytest.raises(TypeError):
        cholesky_solve(A, b)


def test_cholesky_solve_invalid_tolerance_type():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        cholesky_solve(A, b, tolerance="1e-12")


def test_cholesky_solve_boolean_tolerance():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        cholesky_solve(A, b, tolerance=True)


def test_cholesky_solve_invalid_tolerance_value():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(ValueError):
        cholesky_solve(A, b, tolerance=0)

    with pytest.raises(ValueError):
        cholesky_solve(A, b, tolerance=-1e-12)


def test_cholesky_solve_identity_matrix():
    A = np.eye(4)

    b = np.array([
        3.0,
        -2.0,
        7.0,
        4.0
    ])

    x = cholesky_solve(A, b)

    assert x == pytest.approx(b)


# ==============================================================================
# Tests for cholesky_decomposition
# ==============================================================================

def test_cholesky_decomposition_2x2():
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    L = cholesky_decomposition(A, lower=True)

    # L debe ser triangular inferior
    assert np.allclose(np.tril(L), L)

    # La diagonal de L debe ser positiva
    assert np.all(np.diag(L) > 0)

    # A = L @ L.T
    assert np.allclose(L @ L.T, A)


def test_cholesky_decomposition_3x3():
    A = np.array([
        [4.0, 12.0, -16.0],
        [12.0, 37.0, -43.0],
        [-16.0, -43.0, 98.0]
    ])

    L = cholesky_decomposition(A, lower=True)

    assert np.allclose(np.tril(L), L)
    assert np.all(np.diag(L) > 0)
    assert np.allclose(L @ L.T, A)


def test_cholesky_decomposition_upper():
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    U = cholesky_decomposition(A, lower=False)

    # U debe ser triangular superior
    assert np.allclose(np.triu(U), U)

    # A = U.T @ U
    assert np.allclose(U.T @ U, A)


def test_cholesky_decomposition_non_symmetric():
    A = np.array([
        [4.0, 1.0],
        [2.0, 3.0]
    ])

    with pytest.raises(ValueError):
        cholesky_decomposition(A)


def test_cholesky_decomposition_not_positive_definite():
    A = np.array([
        [1.0, 2.0],
        [2.0, 1.0]
    ])

    with pytest.raises(ValueError):
        cholesky_decomposition(A)


def test_cholesky_decomposition_negative_diagonal():
    A = np.array([
        [-4.0, 0.0],
        [0.0, 4.0]
    ])

    with pytest.raises(ValueError):
        cholesky_decomposition(A)


def test_cholesky_decomposition_invalid_inputs():
    with pytest.raises(ValueError):
        cholesky_decomposition(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        cholesky_decomposition(np.array([[1.0, 2.0]]))

    with pytest.raises(TypeError):
        cholesky_decomposition([[1.0, "bad"], [2.0, 3.0]])

    with pytest.raises(TypeError):
        cholesky_decomposition(np.eye(2), tolerance="bad")

    with pytest.raises(TypeError):
        cholesky_decomposition(np.eye(2), tolerance=False)

    with pytest.raises(TypeError):
        cholesky_decomposition(np.eye(2), lower="not_a_bool")


# ==============================================================================
# Tests for cholesky convenience wrapper & aliases
# ==============================================================================

def test_cholesky_convenience_solve():
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    b = np.array([8.0, 7.0])

    x = cholesky(A, b)
    assert x == pytest.approx([1.25, 1.5])


def test_cholesky_convenience_decomposition():
    A = np.array([
        [4.0, 2.0],
        [2.0, 3.0]
    ])

    L = cholesky(A)
    assert np.allclose(L @ L.T, A)

    U = cholesky(A, lower=False)
    assert np.allclose(U.T @ U, A)


def test_cholesky_factorization_alias():
    assert cholesky_factorization is cholesky_decomposition
