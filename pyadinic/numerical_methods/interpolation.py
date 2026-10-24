def linear_interpolation(x, x0, x1, y0, y1):

    mags = [x, x0, x1, y0, y1]

    for mag in mags:
        if not isinstance(mag,(int, float)):
            raise TypeError("No numeric args are not allowed")
        if isinstance(mag, bool):
            raise TypeError("Not booleans allowed")

    if x0 == x1:
        raise ValueError("Zero divition error")
    if x == x0:
        return y0
    if x == x1:
        return y1
    if (x < min(x0, x1)) or (x > max(x0, x1)):
        raise ValueError("Query out of range")

    interp = y0 + (((x - x0)/(x1 - x0)) * (y1 - y0))
    return interp

def lagrange_interpolation(x, x_values, y_values):

    for value in x_values:
        if not isinstance(value, (int, float)):
            raise TypeError("x_values must contain only numeric values")
        if isinstance(value, bool):
            raise TypeError("Booleans are not allowed")

    for value in y_values:
        if not isinstance(value, (int, float)):
            raise TypeError("y_values must contain only numeric values")
        if isinstance(value, bool):
            raise TypeError("Booleans are not allowed")

    if not isinstance(x, (int, float)):
        raise TypeError("x must be numeric")

    if isinstance(x, bool):
        raise TypeError("Booleans are not allowed")

    if len(x_values) != len(y_values):
        raise ValueError("x_values and y_values must have the same length")

    if len(set(x_values)) != len(x_values):
        raise ValueError("x_values cannot contain repeated values")

    if len(x_values) < 2:
        raise ValueError("At least two points are required")
    
    result = 0

    for i in range(len(x_values)):
        term = y_values[i]

        for j in range(len(x_values)):
            if i != j:
                term *= (
                    (x - x_values[j])
                    / (x_values[i] - x_values[j])
                )

        result += term

    return result

def newton_interpolation(x, x_values, y_values):

    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError("x must be numeric")

    for value in x_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("x_values must contain only numeric values")

    for value in y_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
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

    result = newton_coefficients[-1]

    for i in range(len(newton_coefficients) - 2, -1, -1):
        result = (
            newton_coefficients[i]
            + (x - x_values[i]) * result
        )

    return result

def cubic_spline_interpolation(x, x_values, y_values):

    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError("x must be numeric")

    for value in x_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                "x_values must contain only numeric values"
            )

    for value in y_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
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

    if x < x_values[0] or x > x_values[-1]:
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

    interval = 0

    for i in range(n):

        if (
            x_values[i]
            <= x
            <= x_values[i + 1]
        ):
            interval = i
            break

    xi = x_values[interval]
    xi1 = x_values[interval + 1]

    yi = y_values[interval]
    yi1 = y_values[interval + 1]

    mi = m[interval]
    mi1 = m[interval + 1]

    hi = xi1 - xi

    result = (
        mi * (xi1 - x) ** 3 / (6 * hi)
        + mi1 * (x - xi) ** 3 / (6 * hi)
        + (
            yi - mi * hi ** 2 / 6
        ) * (xi1 - x) / hi
        + (
            yi1 - mi1 * hi ** 2 / 6
        ) * (x - xi) / hi
    )

    return result


def pchip_interpolation(x, x_values, y_values): 

    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError("x must be numeric")

    for value in x_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError(
                "x_values must contain only numeric values"
            )

    for value in y_values:
        if not isinstance(value, (int, float)) or isinstance(value, bool):
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

    if x < x_values[0] or x > x_values[-1]:
        raise ValueError(
            "Query out of interpolation range"
        )

    delta = []
    h = []

    for i in range(len(x_values)-1):
        h.append(x_values[i + 1] - x_values[i])
        delta.append((y_values[i + 1] - y_values[i])/h[-1])

    pend_nodes = [0.0] * len(x_values)

    for i in range(1, len(x_values)-1):
        if (delta[i-1] * delta[i]) <= 0:
            pend_nodes[i] = 0
        else:
            w1 = 2 * h[i] + h[i - 1]
            w2 = h[i] + 2 * h[i - 1]

            pend_node = (w1 + w2)/((w1/delta[i-1])+(w2/delta[i])) 
            pend_nodes[i] = pend_node


    pend_nodes[0] = (((2 * h[0] + h[1]) * delta[0] - h[0] * delta[1])/ (h[0] + h[1]))
    pend_nodes[-1] = (((2 * h[-1] + h[-2]) * delta[-1] - h[-1] * delta[-2])/ (h[-1] + h[-2]))

    def H00(t):
        return (2*(t**3) - 3*(t**2) + 1)
    def H10(t):
        return ((t**3) - 2*(t**2) + t)
    def H01(t):
        return (-2*(t**3) +  3*(t**2))
    def H11(t):   
        return ((t**3) - (t**2))

    
    for i in range(len(x_values) - 1):
        if x_values[i] <= x <= x_values[i + 1]:

            y_an = y_values[i]
            y_pos = y_values[i + 1]
            d_an = pend_nodes[i]
            d_pos = pend_nodes[i + 1]
            h_ex = h[i]

            t = (x - x_values[i]) / h_ex

            s = (
                H00(t) * y_an
                + H10(t) * h_ex * d_an
                + H01(t) * y_pos
                + H11(t) * h_ex * d_pos
            )

            return s


