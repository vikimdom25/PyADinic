import numpy as np
import pytest

from pyadinic.numerical_methods.interpolation import (
    linear_interpolation,
    lagrange_interpolation,
    newton_interpolation,
    cubic_spline_interpolation,
    pchip_interpolation,
)


class TestLinearInterpolation:
    def test_scalar_vs_vector_equivalence(self):
        x0, x1 = 0.0, 10.0
        y0, y1 = 5.0, 25.0
        x_queries = [0.0, 2.5, 5.0, 7.5, 10.0]

        scalars = [linear_interpolation(xi, x0, x1, y0, y1) for xi in x_queries]
        vec_list = linear_interpolation(x_queries, x0, x1, y0, y1)
        vec_arr = linear_interpolation(np.array(x_queries), x0, x1, y0, y1)

        assert isinstance(scalars[0], float)
        assert isinstance(vec_list, np.ndarray)
        assert isinstance(vec_arr, np.ndarray)
        assert np.allclose(vec_list, scalars)
        assert np.allclose(vec_arr, scalars)

    def test_out_of_range_vector(self):
        with pytest.raises(ValueError, match="Query out of range"):
            linear_interpolation([1.0, 11.0], 0.0, 10.0, 0.0, 10.0)

        with pytest.raises(ValueError, match="Query out of range"):
            linear_interpolation([-0.1, 5.0], 0.0, 10.0, 0.0, 10.0)

    def test_type_errors(self):
        with pytest.raises(TypeError):
            linear_interpolation([1.0, True], 0.0, 10.0, 0.0, 10.0)
        with pytest.raises(TypeError):
            linear_interpolation([1.0, "str"], 0.0, 10.0, 0.0, 10.0)


class TestLagrangeInterpolation:
    def test_scalar_vs_vector_equivalence(self):
        x_vals = [0.0, 1.0, 2.0, 3.0]
        y_vals = [1.0, 3.0, 2.0, 5.0]
        x_queries = [0.2, 0.8, 1.5, 2.7]

        scalars = [lagrange_interpolation(xi, x_vals, y_vals) for xi in x_queries]
        vec_list = lagrange_interpolation(x_queries, x_vals, y_vals)
        vec_arr = lagrange_interpolation(np.array(x_queries), x_vals, y_vals)

        assert isinstance(scalars[0], float)
        assert isinstance(vec_list, np.ndarray)
        assert isinstance(vec_arr, np.ndarray)
        assert np.allclose(vec_list, scalars)
        assert np.allclose(vec_arr, scalars)

    def test_type_errors(self):
        with pytest.raises(TypeError):
            lagrange_interpolation([1.0, True], [0, 1, 2], [0, 1, 4])
        with pytest.raises(TypeError):
            lagrange_interpolation(["abc"], [0, 1, 2], [0, 1, 4])


class TestNewtonInterpolation:
    def test_scalar_vs_vector_equivalence(self):
        x_vals = [0.0, 1.0, 2.0, 3.0]
        y_vals = [0.0, 1.0, 8.0, 27.0]  # x^3
        x_queries = [0.5, 1.5, 2.5]

        scalars = [newton_interpolation(xi, x_vals, y_vals) for xi in x_queries]
        vec_list = newton_interpolation(x_queries, x_vals, y_vals)
        vec_arr = newton_interpolation(np.array(x_queries), x_vals, y_vals)

        assert isinstance(scalars[0], float)
        assert isinstance(vec_list, np.ndarray)
        assert isinstance(vec_arr, np.ndarray)
        assert np.allclose(vec_list, scalars)
        assert np.allclose(vec_arr, scalars)
        assert np.allclose(vec_arr, [0.5**3, 1.5**3, 2.5**3])

    def test_type_errors(self):
        with pytest.raises(TypeError):
            newton_interpolation([1.0, True], [0, 1, 2], [0, 1, 4])
        with pytest.raises(TypeError):
            newton_interpolation("bad", [0, 1, 2], [0, 1, 4])


class TestCubicSplineInterpolation:
    def test_scalar_vs_vector_equivalence(self):
        x_vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_vals = [0.0, 1.0, 4.0, 9.0, 16.0]
        x_queries = [0.25, 1.5, 2.75, 3.5]

        scalars = [cubic_spline_interpolation(xi, x_vals, y_vals) for xi in x_queries]
        vec_list = cubic_spline_interpolation(x_queries, x_vals, y_vals)
        vec_arr = cubic_spline_interpolation(np.array(x_queries), x_vals, y_vals)

        assert isinstance(scalars[0], float)
        assert isinstance(vec_list, np.ndarray)
        assert isinstance(vec_arr, np.ndarray)
        assert np.allclose(vec_list, scalars)
        assert np.allclose(vec_arr, scalars)

    def test_out_of_range_vector(self):
        x_vals = [0.0, 1.0, 2.0]
        y_vals = [0.0, 1.0, 4.0]
        with pytest.raises(ValueError, match="Query out of interpolation range"):
            cubic_spline_interpolation([-0.1, 1.0], x_vals, y_vals)
        with pytest.raises(ValueError, match="Query out of interpolation range"):
            cubic_spline_interpolation([0.5, 2.1], x_vals, y_vals)

    def test_type_errors(self):
        x_vals = [0.0, 1.0, 2.0]
        y_vals = [0.0, 1.0, 4.0]
        with pytest.raises(TypeError):
            cubic_spline_interpolation([0.5, True], x_vals, y_vals)


class TestPchipInterpolation:
    def test_scalar_vs_vector_equivalence(self):
        x_vals = [0.0, 1.0, 2.0, 3.0, 4.0]
        y_vals = [0.0, 1.0, 2.0, 2.0, 3.0]
        x_queries = [0.3, 1.2, 2.5, 3.8]

        scalars = [pchip_interpolation(xi, x_vals, y_vals) for xi in x_queries]
        vec_list = pchip_interpolation(x_queries, x_vals, y_vals)
        vec_arr = pchip_interpolation(np.array(x_queries), x_vals, y_vals)

        assert isinstance(scalars[0], float)
        assert isinstance(vec_list, np.ndarray)
        assert isinstance(vec_arr, np.ndarray)
        assert np.allclose(vec_list, scalars)
        assert np.allclose(vec_arr, scalars)

    def test_out_of_range_vector(self):
        x_vals = [0.0, 1.0, 2.0]
        y_vals = [0.0, 1.0, 4.0]
        with pytest.raises(ValueError, match="Query out of interpolation range"):
            pchip_interpolation([-0.01, 0.5], x_vals, y_vals)
        with pytest.raises(ValueError, match="Query out of interpolation range"):
            pchip_interpolation([0.5, 2.01], x_vals, y_vals)

    def test_type_errors(self):
        x_vals = [0.0, 1.0, 2.0]
        y_vals = [0.0, 1.0, 4.0]
        with pytest.raises(TypeError):
            pchip_interpolation([0.5, True], x_vals, y_vals)
