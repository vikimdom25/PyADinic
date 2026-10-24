import numpy as np
import pytest

from pyadinic.numerical_methods.linear_systems import gaussian_elimination, gauss_jordan


def test_gaussian_elimination_2x2():
    A = np.array([
        [2.0, 1.0],
        [1.0, -1.0]
    ])

    b = np.array([
        3.0,
        0.0
    ])

    x = gaussian_elimination(A, b)

    expected = np.array([
        1.0,
        1.0
    ])

    assert x == pytest.approx(expected)


def test_gaussian_elimination_3x3():
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

    x = gaussian_elimination(A, b)

    expected = np.array([
        1.0,
        -2.0,
        -2.0
    ])

    assert x == pytest.approx(expected)


def test_gaussian_elimination_partial_pivoting():
    A = np.array([
        [0.0, 2.0],
        [1.0, 3.0]
    ])

    b = np.array([
        4.0,
        5.0
    ])

    x = gaussian_elimination(A, b)

    expected = np.array([
        -1.0,
        2.0
    ])

    assert x == pytest.approx(expected)


def test_gaussian_elimination_accepts_lists():
    A = [
        [2.0, 1.0],
        [1.0, -1.0]
    ]

    b = [
        3.0,
        0.0
    ]

    x = gaussian_elimination(A, b)

    expected = np.array([
        1.0,
        1.0
    ])

    assert x == pytest.approx(expected)


def test_gaussian_elimination_does_not_modify_inputs():
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

    gaussian_elimination(A, b)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)


def test_gaussian_elimination_singular_matrix():
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])

    b = np.array([
        3.0,
        6.0
    ])

    with pytest.raises(ValueError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_nearly_singular_matrix():
    A = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-14]
    ])

    b = np.array([
        2.0,
        2.0
    ])

    with pytest.raises(ValueError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_invalid_A_dimension():
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_invalid_A_not_2d():
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
        gaussian_elimination(A, b)


def test_gaussian_elimination_invalid_b_dimension():
    A = np.eye(2)

    b = np.array([
        [1.0],
        [2.0]
    ])

    with pytest.raises(ValueError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_incompatible_dimensions():
    A = np.eye(3)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_invalid_A_values():
    A = np.array([
        [1.0, 2.0],
        [3.0, "text"]
    ], dtype=object)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(TypeError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_invalid_b_values():
    A = np.eye(2)

    b = np.array([
        1.0,
        "text"
    ], dtype=object)

    with pytest.raises(TypeError):
        gaussian_elimination(A, b)


def test_gaussian_elimination_invalid_tolerance_type():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        gaussian_elimination(A, b, tolerance="1e-12")


def test_gaussian_elimination_boolean_tolerance():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        gaussian_elimination(A, b, tolerance=True)


def test_gaussian_elimination_invalid_tolerance_value():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(ValueError):
        gaussian_elimination(A, b, tolerance=0)

    with pytest.raises(ValueError):
        gaussian_elimination(A, b, tolerance=-1e-12)


def test_gaussian_elimination_identity_matrix():
    A = np.eye(4)

    b = np.array([
        3.0,
        -2.0,
        7.0,
        4.0
    ])

    x = gaussian_elimination(A, b)

    assert x == pytest.approx(b)


def test_gauss_jordan_2x2():
    A = np.array([
        [2.0, 1.0],
        [1.0, -1.0]
    ])

    b = np.array([
        3.0,
        0.0
    ])

    x = gauss_jordan(A, b)

    expected = np.array([
        1.0,
        1.0
    ])

    assert x == pytest.approx(expected)


def test_gauss_jordan_3x3():
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

    x = gauss_jordan(A, b)

    expected = np.array([
        1.0,
        -2.0,
        -2.0
    ])

    assert x == pytest.approx(expected)


def test_gauss_jordan_partial_pivoting():
    A = np.array([
        [0.0, 2.0],
        [1.0, 3.0]
    ])

    b = np.array([
        4.0,
        5.0
    ])

    x = gauss_jordan(A, b)

    expected = np.array([
        -1.0,
        2.0
    ])

    assert x == pytest.approx(expected)


def test_gauss_jordan_accepts_lists():
    A = [
        [2.0, 1.0],
        [1.0, -1.0]
    ]

    b = [
        3.0,
        0.0
    ]

    x = gauss_jordan(A, b)

    expected = np.array([
        1.0,
        1.0
    ])

    assert x == pytest.approx(expected)


def test_gauss_jordan_does_not_modify_inputs():
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

    gauss_jordan(A, b)

    assert np.array_equal(A, A_original)
    assert np.array_equal(b, b_original)


def test_gauss_jordan_singular_matrix():
    A = np.array([
        [1.0, 2.0],
        [2.0, 4.0]
    ])

    b = np.array([
        3.0,
        6.0
    ])

    with pytest.raises(ValueError):
        gauss_jordan(A, b)


def test_gauss_jordan_nearly_singular_matrix():
    A = np.array([
        [1.0, 1.0],
        [1.0, 1.0 + 1e-14]
    ])

    b = np.array([
        2.0,
        2.0
    ])

    with pytest.raises(ValueError):
        gauss_jordan(A, b)


def test_gauss_jordan_invalid_A_dimension():
    A = np.array([
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0]
    ])

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        gauss_jordan(A, b)


def test_gauss_jordan_invalid_A_not_2d():
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
        gauss_jordan(A, b)


def test_gauss_jordan_invalid_b_dimension():
    A = np.eye(2)

    b = np.array([
        [1.0],
        [2.0]
    ])

    with pytest.raises(ValueError):
        gauss_jordan(A, b)


def test_gauss_jordan_incompatible_dimensions():
    A = np.eye(3)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(ValueError):
        gauss_jordan(A, b)


def test_gauss_jordan_invalid_A_values():
    A = np.array([
        [1.0, 2.0],
        [3.0, "text"]
    ], dtype=object)

    b = np.array([
        1.0,
        2.0
    ])

    with pytest.raises(TypeError):
        gauss_jordan(A, b)


def test_gauss_jordan_invalid_b_values():
    A = np.eye(2)

    b = np.array([
        1.0,
        "text"
    ], dtype=object)

    with pytest.raises(TypeError):
        gauss_jordan(A, b)


def test_gauss_jordan_invalid_tolerance_type():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        gauss_jordan(A, b, tolerance="1e-12")


def test_gauss_jordan_boolean_tolerance():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(TypeError):
        gauss_jordan(A, b, tolerance=True)


def test_gauss_jordan_invalid_tolerance_value():
    A = np.eye(2)
    b = np.ones(2)

    with pytest.raises(ValueError):
        gauss_jordan(A, b, tolerance=0)

    with pytest.raises(ValueError):
        gauss_jordan(A, b, tolerance=-1e-12)


def test_gauss_jordan_identity_matrix():
    A = np.eye(4)

    b = np.array([
        3.0,
        -2.0,
        7.0,
        4.0
    ])

    x = gauss_jordan(A, b)

    assert x == pytest.approx(b)