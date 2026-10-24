import numpy as np


def _parse_query_x(x, bool_error_msg, non_numeric_error_msg):
    if isinstance(x, (bool, np.bool_)):
        raise TypeError(bool_error_msg)
    if isinstance(x, (int, float, np.integer, np.floating)):
        return x, False
    if isinstance(x, (list, tuple, np.ndarray)):
        if isinstance(x, np.ndarray) and (x.dtype.kind == 'b' or x.dtype == bool):
            raise TypeError(bool_error_msg)

        def _check_items(items):
            for item in items:
                if isinstance(item, (bool, np.bool_)):
                    raise TypeError(bool_error_msg)
                elif isinstance(item, (list, tuple)):
                    _check_items(item)
                elif not isinstance(item, (int, float, np.integer, np.floating)):
                    raise TypeError(non_numeric_error_msg)

        _check_items(x if not isinstance(x, np.ndarray) else x.flat)
        arr = np.asarray(x, dtype=float)
        return arr, True
    raise TypeError(non_numeric_error_msg)


def linear_interpolation(x, x0, x1, y0, y1):
    mags = [x0, x1, y0, y1]

    for mag in mags:
        if isinstance(mag, (bool, np.bool_)):
            raise TypeError("Not booleans allowed")
        if not isinstance(mag, (int, float, np.integer, np.floating)):
            raise TypeError("No numeric args are not allowed")

    if x0 == x1:
        raise ValueError("Zero divition error")

    x_val, is_vec = _parse_query_x(x, "Not booleans allowed", "No numeric args are not allowed")

    if not is_vec:
        if (x_val < min(x0, x1)) or (x_val > max(x0, x1)):
            raise ValueError("Query out of range")
        if x_val == x0:
            return y0
        if x_val == x1:
            return y1
        interp = y0 + (((x_val - x0) / (x1 - x0)) * (y1 - y0))
        return interp
    else:
        if np.any(x_val < min(x0, x1)) or np.any(x_val > max(x0, x1)):
            raise ValueError("Query out of range")
        return y0 + (((x_val - x0) / (x1 - x0)) * (y1 - y0))


def lagrange_interpolation(x, x_values, y_values):
    for value in x_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Booleans are not allowed")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("x_values must contain only numeric values")

    for value in y_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("Booleans are not allowed")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("y_values must contain only numeric values")

    x_val, is_vec = _parse_query_x(x, "Booleans are not allowed", "x must be numeric")

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")

    if len(set(x_values)) != len(x_values):
        raise ValueError("x_values cannot contain repeated values")

    if len(x_values) < 2:
        raise ValueError("At least two points are required")

    def _eval(xi):
        result = 0.0
        for i in range(len(x_values)):
            term = y_values[i]
            for j in range(len(x_values)):
                if i != j:
                    term *= (
                        (xi - x_values[j])
                        / (x_values[i] - x_values[j])
                    )
            result += term
        return result

    if not is_vec:
        return _eval(x_val)
    else:
        return np.vectorize(_eval, otypes=[float])(x_val)


def newton_interpolation(x, x_values, y_values):
    x_val, is_vec = _parse_query_x(x, "x must be numeric", "x must be numeric")

    for value in x_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("x_values must contain only numeric values")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("x_values must contain only numeric values")

    for value in y_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("y_values must contain only numeric values")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError("y_values must contain only numeric values")

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")

    if len(x_values) < 2:
        raise ValueError("At least two points are required")

    if len(set(x_values)) != len(x_values):
        raise ValueError("x_values cannot contain repeated values")

    coefficients = [y for y in y_values]
    newton_coefficients = [coefficients[0]]

    while len(coefficients) > 1:
        next_level = []
        for i in range(len(coefficients) - 1):
            difference = (
                coefficients[i + 1] - coefficients[i]
            ) / (
                x_values[i + len(y_values) - len(coefficients) + 1]
                - x_values[i]
            )
            next_level.append(difference)

        coefficients = next_level
        newton_coefficients.append(coefficients[0])

    def _eval(xi):
        result = newton_coefficients[-1]
        for i in range(len(newton_coefficients) - 2, -1, -1):
            result = (
                newton_coefficients[i]
                + (xi - x_values[i]) * result
            )
        return float(result)

    if not is_vec:
        return _eval(x_val)
    else:
        return np.vectorize(_eval, otypes=[float])(x_val)


def cubic_spline_interpolation(x, x_values, y_values):
    x_val, is_vec = _parse_query_x(x, "x must be numeric", "x must be numeric")

    for value in x_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("x_values must contain only numeric values")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                "x_values must contain only numeric values"
            )

    for value in y_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("y_values must contain only numeric values")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                "y_values must contain only numeric values"
            )

    if len(x_values) != len(y_values):
        raise ValueError(
            "x_values and y_values must have the same length"
        )

    if len(x_values) < 3:
        raise ValueError(
            "At least three points are required"
        )

    if len(set(x_values)) != len(x_values):
        raise ValueError(
            "x_values cannot contain repeated values"
        )

    if any(x_values[i + 1] <= x_values[i] for i in range(len(x_values) - 1)):
        raise ValueError(
            "x_values must be strictly monotonically increasing"
        )

    if not is_vec:
        if x_val < x_values[0] or x_val > x_values[-1]:
            raise ValueError(
                "Query out of interpolation range"
            )
    else:
        if np.any(x_val < x_values[0]) or np.any(x_val > x_values[-1]):
            raise ValueError(
                "Query out of interpolation range"
            )

    n = len(x_values) - 1

    h = [
        x_values[i + 1] - x_values[i]
        for i in range(n)
    ]

    lower = [0.0] * (n - 2)
    diagonal = [0.0] * (n - 1)
    upper = [0.0] * (n - 2)
    right_side = [0.0] * (n - 1)

    for i in range(1, n):
        diagonal[i - 1] = 2 * (
            h[i - 1] + h[i]
        )

        right_side[i - 1] = 6 * (
            (y_values[i + 1] - y_values[i]) / h[i]
            -
            (y_values[i] - y_values[i - 1]) / h[i - 1]
        )

        if i > 1:
            lower[i - 2] = h[i - 1]

        if i < n - 1:
            upper[i - 1] = h[i]

    from pyadinic.numerical_methods.solvers import (
        solve_tridiagonal
    )

    interior_m = solve_tridiagonal(
        lower,
        diagonal,
        upper,
        right_side
    )

    m = [0.0] + interior_m + [0.0]

    def _eval(xi):
        interval = 0
        for i in range(n):
            if (
                x_values[i]
                <= xi
                <= x_values[i + 1]
            ):
                interval = i
                break

        xi_val = x_values[interval]
        xi1 = x_values[interval + 1]

        yi = y_values[interval]
        yi1 = y_values[interval + 1]

        mi = m[interval]
        mi1 = m[interval + 1]

        hi = xi1 - xi_val

        result = (
            mi * (xi1 - xi) ** 3 / (6 * hi)
            + mi1 * (xi - xi_val) ** 3 / (6 * hi)
            + (
                yi - mi * hi ** 2 / 6
            ) * (xi1 - xi) / hi
            + (
                yi1 - mi1 * hi ** 2 / 6
            ) * (xi - xi_val) / hi
        )
        return float(result)

    if not is_vec:
        return _eval(x_val)
    else:
        return np.vectorize(_eval, otypes=[float])(x_val)


def pchip_interpolation(x, x_values, y_values): 
    x_val, is_vec = _parse_query_x(x, "x must be numeric", "x must be numeric")

    for value in x_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("x_values must contain only numeric values")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                "x_values must contain only numeric values"
            )

    for value in y_values:
        if isinstance(value, (bool, np.bool_)):
            raise TypeError("y_values must contain only numeric values")
        if not isinstance(value, (int, float, np.integer, np.floating)):
            raise TypeError(
                "y_values must contain only numeric values"
            )

    if len(x_values) != len(y_values):
        raise ValueError(
            "x_values and y_values must have the same length"
        )

    if len(x_values) < 3:
        raise ValueError(
            "At least three points are required"
        )

    if len(set(x_values)) != len(x_values):
        raise ValueError(
            "x_values cannot contain repeated values"
        )

    if any(x_values[i + 1] <= x_values[i] for i in range(len(x_values) - 1)):
        raise ValueError(
            "x_values must be strictly monotonically increasing"
        )

    if not is_vec:
        if x_val < x_values[0] or x_val > x_values[-1]:
            raise ValueError(
                "Query out of interpolation range"
            )
    else:
        if np.any(x_val < x_values[0]) or np.any(x_val > x_values[-1]):
            raise ValueError(
                "Query out of interpolation range"
            )

    delta = []
    h = []

    for i in range(len(x_values) - 1):
        h.append(x_values[i + 1] - x_values[i])
        delta.append((y_values[i + 1] - y_values[i]) / h[-1])

    pend_nodes = [0.0] * len(x_values)

    for i in range(1, len(x_values) - 1):
        if (delta[i - 1] * delta[i]) <= 0:
            pend_nodes[i] = 0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]

            pend_node = (w1 + w2) / ((w1 / delta[i - 1]) + (w2 / delta[i])) 
            pend_nodes[i] = pend_node

    pend_nodes[0] = (((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1]) / (h[0] + h[1]))
    pend_nodes[-1] = (((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2]) / (h[-1] + h[-2]))

    def H00(t):
        return (2 * (t ** 3) - 3 * (t ** 2) + 1)
    def H10(t):
        return ((t ** 3) - 2 * (t ** 2) + t)
    def H01(t):
        return (-2 * (t ** 3) + 3 * (t ** 2))
    def H11(t):   
        return ((t ** 3) - (t ** 2))

    def _eval(xi):
        for i in range(len(x_values) - 1):
            if x_values[i] <= xi <= x_values[i + 1]:
                y_an = y_values[i]
                y_pos = y_values[i + 1]
                d_an = pend_nodes[i]
                d_pos = pend_nodes[i + 1]
                h_ex = h[i]

                t = (xi - x_values[i]) / h_ex

                s = (
                    H00(t) * y_an
                    + H10(t) * h_ex * d_an
                    + H01(t) * y_pos
                    + H11(t) * h_ex * d_pos
                )

                return float(s)
        return float(y_values[-1])

    if not is_vec:
        return _eval(x_val)
    else:
        return np.vectorize(_eval, otypes=[float])(x_val)
