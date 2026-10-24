import numpy as np
import pytest

from pyadinic.numerical_methods.linear_systems import (
    lu_decomposition,
    lu_solve,
    lu,
    lu_factorization,
)


# ==============================================================================
# Tests for lu_solve
# ==============================================================================

def test_lu_solve_2x2():
    A = np.array([
        [2.0, 1.0],
        [1.0, -1.0]
    ])

    b = np.array([
        3.0,
        0.0
    ])

    x = lu_solve(A, b)

    expected = np.array([
        1.0,
        1.0
    ])

    assert x == pytest.approx(expected)


def test_lu_solve_3x3():
    A = np.array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 4.0],
        [-1.0, 0.5, -1.0]
    ])

    b = np.array([
        1.0,
        -2.0,
        0.0
    ])

    x = lu_solve(A, b)

    expected = np.array([
        1.0,
        -2.0,
        -2.0
    ])

    assert x == pytest.approx(expected)


def test_lu_solve_partial_pivoting():
    A = np.array([
        [0.0, 2.0],
        [1.0, 3.0]
    ])

    b = np.array([
        4.0,
        5.0
    ])

    x = lu_solve(A, b)

    expected = np.array([
        -1.0,
        2.0
    ])

    assert x == pytest.approx(expected)


def test_lu_solve_accepts_lists():
    A = [
        [2.0, 1.0],
        [1.0, -1.0]
    ]

    b = [
        3.0,
        0.0
    ]

    x = lu_solve(A, b)

    expected = np.array([
        1.0,
        1.0
    ])

    assert x == pytest.approx(expected)


def test_lu_solve_does_not_modify_inputs():
    A = np.array([
        [0.0, 2.0],
        [1.0, 3.0]
    ])

    b = np.array([
        4.0,
        5.0
    ])

    A_original = A.copy()
    b_original = b.copy()

    lu_solve(A, b)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)


def test_lu_solve_singular_matrix():
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])

    b = np.array([
        3.0,
        6.0
    ])

    with pytest.raises(ValueError):
        lu_solve(A, b)


def test_lu_solve_nearly_singular_matrix():
    A = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-14]
    ])

    b = np.array([
        2.0,
        2.0
    ])

    with pytest.raises(ValueError):
        lu_solve(A, b)


def test_lu_solve_invalid_A_dimension():
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        lu_solve(A, b)


def test_lu_solve_invalid_A_not_2d():
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
        lu_solve(A, b)


def test_lu_solve_invalid_b_dimension():
    A = np.eye(2)

    b = np.array([
        [1.0],
        [2.0]
    ])

    with pytest.raises(ValueError):
        lu_solve(A, b)


def test_lu_solve_incompatible_dimensions():
    A = np.eye(3)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        lu_solve(A, b)


def test_lu_solve_invalid_A_values():
    A = np.array([
        [1.0, 2.0],
        [3.0, "text"]
    ], dtype=object)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(TypeError):
        lu_solve(A, b)


def test_lu_solve_invalid_b_values():
    A = np.eye(2)

    b = np.array([
        1.0,
        "text"
    ], dtype=object)

    with pytest.raises(TypeError):
        lu_solve(A, b)


def test_lu_solve_invalid_tolerance_type():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        lu_solve(A, b, tolerance="1e-12")


def test_lu_solve_boolean_tolerance():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        lu_solve(A, b, tolerance=True)


def test_lu_solve_invalid_tolerance_value():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(ValueError):
        lu_solve(A, b, tolerance=0)

    with pytest.raises(ValueError):
        lu_solve(A, b, tolerance=-1e-12)


def test_lu_solve_identity_matrix():
    A = np.eye(4)

    b = np.array([
        3.0,
        -2.0,
        7.0,
        4.0
    ])

    x = lu_solve(A, b)

    assert x == pytest.approx(b)


# ==============================================================================
# Tests for lu_decomposition
# ==============================================================================

def test_lu_decomposition_without_pivoting_2x2():
    A = np.array([
        [2.0, 1.0],
        [1.0, -1.0]
    ])

    L, U = lu_decomposition(A, pivoting=False)

    # L debe ser triangular inferior unitaria
    assert np.allclose(np.tril(L), L)
    assert np.allclose(np.diag(L), 1.0)

    # U debe ser triangular superior
    assert np.allclose(np.triu(U), U)

    # A = L @ U
    assert np.allclose(L @ U, A)


def test_lu_decomposition_without_pivoting_3x3():
    A = np.array([
        [3.0, 2.0, -1.0],
        [2.0, -2.0, 4.0],
        [-1.0, 0.5, -1.0]
    ])

    L, U = lu_decomposition(A, pivoting=False)

    assert np.allclose(np.tril(L), L)
    assert np.allclose(np.diag(L), 1.0)
    assert np.allclose(np.triu(U), U)
    assert np.allclose(L @ U, A)


def test_lu_decomposition_with_pivoting():
    A = np.array([
        [0.0, 2.0],
        [1.0, 3.0]
    ])

    P, L, U = lu_decomposition(A, pivoting=True)

    # P debe ser matriz de permutación
    assert np.allclose(P @ P.T, np.eye(2))

    # L debe ser triangular inferior unitaria
    assert np.allclose(np.tril(L), L)
    assert np.allclose(np.diag(L), 1.0)

    # U debe ser triangular superior
    assert np.allclose(np.triu(U), U)

    # P @ A = L @ U
    assert np.allclose(P @ A, L @ U)


def test_lu_decomposition_singular():
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])

    with pytest.raises(ValueError):
        lu_decomposition(A, pivoting=False)

    with pytest.raises(ValueError):
        lu_decomposition(A, pivoting=True)


def test_lu_decomposition_invalid_inputs():
    with pytest.raises(ValueError):
        lu_decomposition(np.array([1.0, 2.0]))

    with pytest.raises(ValueError):
        lu_decomposition(np.array([[1.0, 2.0]]))

    with pytest.raises(TypeError):
        lu_decomposition([[1.0, "invalid"], [2.0, 3.0]])

    with pytest.raises(TypeError):
        lu_decomposition(np.eye(2), tolerance="bad")

    with pytest.raises(TypeError):
        lu_decomposition(np.eye(2), tolerance=False)

    with pytest.raises(TypeError):
        lu_decomposition(np.eye(2), pivoting="not_a_bool")


# ==============================================================================
# Tests for lu convenience wrapper & aliases
# ==============================================================================

def test_lu_convenience_solve():
    A = np.array([
        [2.0, 1.0],
        [1.0, -1.0]
    ])

    b = np.array([3.0, 0.0])

    x = lu(A, b)
    assert x == pytest.approx([1.0, 1.0])


def test_lu_convenience_decomposition():
    A = np.array([
        [2.0, 1.0],
        [1.0, -1.0]
    ])

    L, U = lu(A)
    assert np.allclose(L @ U, A)

    P, L, U = lu(A, pivoting=True)
    assert np.allclose(P @ A, L @ U)


def test_lu_factorization_alias():
    assert lu_factorization is lu_decomposition
