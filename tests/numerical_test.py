import pytest

from pyadinic.numerical_methods import numerical_derivative, solve_secant, solve_bisection


def test_numerical_derivative_quadratic():
    def function(x):
        return x**2

    result = numerical_derivative(function, 2.0)

    assert result == pytest.approx(4.0, rel=1e-6)


def test_numerical_derivative_cubic():
    def function(x):
        return x**3

    result = numerical_derivative(function, 2.0)

    assert result == pytest.approx(12.0, rel=1e-6)


@pytest.mark.parametrize("x", [-3.0, -1.0, 0.0, 1.0, 3.0])
def test_numerical_derivative_quadratic_multiple_points(x):
    def function(x):
        return x**2

    result = numerical_derivative(function, x)

    assert result == pytest.approx(2 * x, rel=1e-6)


def test_numerical_derivative_custom_step():
    def function(x):
        return x**2

    result = numerical_derivative(function, 2.0, h=1e-5)

    assert result == pytest.approx(4.0, rel=1e-5)


def test_numerical_derivative_function_not_callable():
    with pytest.raises(TypeError):
        numerical_derivative(10, 2.0)


@pytest.mark.parametrize("x", ["2", None, [2]])
def test_numerical_derivative_x_not_numeric(x):
    def function(x):
        return x**2

    with pytest.raises(TypeError):
        numerical_derivative(function, x)


@pytest.mark.parametrize("h", ["1e-6", None, [1e-6], 0, -1e-6])
def test_numerical_derivative_invalid_h(h):
    def function(x):
        return x**2

    if isinstance(h, (int, float)) and not isinstance(h, bool) and h <= 0:
        expected = ValueError
    else:
        expected = TypeError

    with pytest.raises(expected):
        numerical_derivative(function, 2.0, h=h)


def test_numerical_derivative_rejects_boolean_x():
    def function(x):
        return x**2

    with pytest.raises(TypeError):
        numerical_derivative(function, True)


def test_numerical_derivative_rejects_boolean_h():
    def function(x):
        return x**2

    with pytest.raises(TypeError):
        numerical_derivative(function, 2.0, h=True)

from pyadinic.numerical_methods import solve_newton


def test_solve_newton_quadratic():
    def function(x):
        return x**2 - 2

    result = solve_newton(function, 1.0)

    assert result == pytest.approx(2**0.5, rel=1e-7)


def test_solve_newton_cubic():
    def function(x):
        return x**3 - 2

    result = solve_newton(function, 1.0)

    assert result == pytest.approx(2**(1 / 3), rel=1e-7)


@pytest.mark.parametrize(
    "function, x0, expected",
    [
        (lambda x: x**2 - 4, 3.0, 2.0),
        (lambda x: x**2 - 9, 2.0, 3.0),
        (lambda x: x**3 - 8, 1.0, 2.0),
    ],
)
def test_solve_newton_multiple_functions(function, x0, expected):
    result = solve_newton(function, x0)

    assert result == pytest.approx(expected, rel=1e-7)


def test_solve_newton_function_not_callable():
    with pytest.raises(TypeError):
        solve_newton(10, 1.0)


@pytest.mark.parametrize("x0", ["1", None, [1]])
def test_solve_newton_x0_not_numeric(x0):
    def function(x):
        return x**2 - 2

    with pytest.raises(TypeError):
        solve_newton(function, x0)


def test_solve_newton_rejects_boolean_x0():
    def function(x):
        return x**2 - 2

    with pytest.raises(TypeError):
        solve_newton(function, True)


def test_solve_newton_invalid_tolerance():
    def function(x):
        return x**2 - 2

    with pytest.raises(ValueError):
        solve_newton(function, 1.0, tolerance=0)

def test_solve_newton_invalid_max_iterations():
    def function(x):
        return x**2 - 2

    with pytest.raises(ValueError):
        solve_newton(function, 1.0, max_iterations=0)


def test_solve_newton_zero_derivative():
    def function(x):
        return x**2 + 1

    with pytest.raises(ValueError):
        solve_newton(function, 0.0)


def test_solve_newton_max_iterations():
    def function(x):
        return x**2 - 2

    with pytest.raises(RuntimeError):
        solve_newton(function, 1.0, max_iterations=1)


# ============================================================
# SECANT METHOD
# ============================================================

def test_solve_secant_quadratic():
    function = lambda x: x**2 - 2

    result = solve_secant(
        function,
        1,
        2
    )

    assert result == pytest.approx(2**0.5)


def test_solve_secant_cubic():
    function = lambda x: x**3 - 2

    result = solve_secant(
        function,
        1,
        2
    )

    assert result == pytest.approx(2**(1/3))


@pytest.mark.parametrize(
    "function, x0, x1, expected",
    [
        (lambda x: x**2 - 4, 1, 3, 2),
        (lambda x: x**2 - 9, 1, 4, 3),
        (lambda x: x**3 - 8, 1, 3, 2),
        (lambda x: x**3 - 27, 2, 4, 3),
    ]
)
def test_solve_secant_multiple_functions(function, x0, x1, expected):

    result = solve_secant(
        function,
        x0,
        x1
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "function, x0, x1",
    [
        (None, 1, 2),
        (5, 1, 2),
        ("function", 1, 2),
    ]
)
def test_solve_secant_invalid_function(function, x0, x1):

    with pytest.raises(TypeError):
        solve_secant(function, x0, x1)


@pytest.mark.parametrize(
    "x0, x1",
    [
        ("1", 2),
        (1, "2"),
        (None, 2),
        (1, None),
        (True, 2),
        (1, False),
    ]
)
def test_solve_secant_invalid_initial_values(x0, x1):

    with pytest.raises(TypeError):
        solve_secant(
            lambda x: x**2 - 2,
            x0,
            x1
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        0,
        -1,
        -1e-8,
    ]
)
def test_solve_secant_invalid_tolerance(tolerance):

    with pytest.raises(ValueError):
        solve_secant(
            lambda x: x**2 - 2,
            1,
            2,
            tolerance=tolerance
        )


@pytest.mark.parametrize(
    "max_iterations",
    [
        0,
        -1,
        -100,
    ]
)
def test_solve_secant_invalid_max_iterations(max_iterations):

    with pytest.raises(ValueError):
        solve_secant(
            lambda x: x**2 - 2,
            1,
            2,
            max_iterations=max_iterations
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_solve_secant_boolean_arguments(value):

    with pytest.raises(TypeError):
        solve_secant(
            lambda x: x**2 - 2,
            value,
            2
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_solve_secant_boolean_tolerance(value):

    with pytest.raises(TypeError):
        solve_secant(
            lambda x: x**2 - 2,
            1,
            2,
            tolerance=value
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_solve_secant_boolean_max_iterations(value):

    with pytest.raises(TypeError):
        solve_secant(
            lambda x: x**2 - 2,
            1,
            2,
            max_iterations=value
        )


def test_solve_secant_zero_secant_denominator():

    function = lambda x: x**2 + 1

    with pytest.raises(ValueError):
        solve_secant(
            function,
            -1,
            1
        )


def test_solve_secant_max_iterations():

    function = lambda x: x**2 - 2

    with pytest.raises(RuntimeError):
        solve_secant(
            function,
            1,
            2,
            tolerance=1e-15,
            max_iterations=1
        )

# ============================================================
# BISECTION METHOD
# ============================================================

from pyadinic.numerical_methods import solve_bisection


def test_solve_bisection_quadratic():
    function = lambda x: x**2 - 2

    result = solve_bisection(
        function,
        1,
        2
    )

    assert result == pytest.approx(2**0.5)


def test_solve_bisection_cubic():
    function = lambda x: x**3 - 2

    result = solve_bisection(
        function,
        1,
        2
    )

    assert result == pytest.approx(2**(1/3))


@pytest.mark.parametrize(
    "function, lb, up, expected",
    [
        (lambda x: x**2 - 4, 0, 3, 2),
        (lambda x: x**2 - 9, 0, 4, 3),
        (lambda x: x**3 - 8, 0, 3, 2),
        (lambda x: x**3 - 27, 0, 4, 3),
    ]
)
def test_solve_bisection_multiple_functions(function, lb, up, expected):

    result = solve_bisection(
        function,
        lb,
        up
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "function, lb, up",
    [
        (None, 1, 2),
        (5, 1, 2),
        ("function", 1, 2),
    ]
)
def test_solve_bisection_invalid_function(function, lb, up):

    with pytest.raises(TypeError):
        solve_bisection(function, lb, up)


@pytest.mark.parametrize(
    "lb, up",
    [
        ("1", 2),
        (1, "2"),
        (None, 2),
        (1, None),
        (True, 2),
        (1, False),
    ]
)
def test_solve_bisection_invalid_bounds(lb, up):

    with pytest.raises(TypeError):
        solve_bisection(
            lambda x: x**2 - 2,
            lb,
            up
        )


@pytest.mark.parametrize(
    "tolerance",
    [
        0,
        -1,
        -1e-8,
    ]
)
def test_solve_bisection_invalid_tolerance(tolerance):

    with pytest.raises(ValueError):
        solve_bisection(
            lambda x: x**2 - 2,
            1,
            2,
            tolerance=tolerance
        )


@pytest.mark.parametrize(
    "max_iterations",
    [
        0,
        -1,
        -100,
    ]
)
def test_solve_bisection_invalid_max_iterations(max_iterations):

    with pytest.raises(ValueError):
        solve_bisection(
            lambda x: x**2 - 2,
            1,
            2,
            max_iterations=max_iterations
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_solve_bisection_boolean_arguments(value):

    with pytest.raises(TypeError):
        solve_bisection(
            lambda x: x**2 - 2,
            value,
            2
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_solve_bisection_boolean_tolerance(value):

    with pytest.raises(TypeError):
        solve_bisection(
            lambda x: x**2 - 2,
            1,
            2,
            tolerance=value
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_solve_bisection_boolean_max_iterations(value):

    with pytest.raises(TypeError):
        solve_bisection(
            lambda x: x**2 - 2,
            1,
            2,
            max_iterations=value
        )


def test_solve_bisection_no_bracket():

    function = lambda x: x**2 + 1

    with pytest.raises(ValueError):
        solve_bisection(
            function,
            -1,
            1
        )


def test_solve_bisection_root_at_lower_bound():

    function = lambda x: x**2 - 4

    result = solve_bisection(
        function,
        2,
        5
    )

    assert result == pytest.approx(2)


def test_solve_bisection_root_at_upper_bound():

    function = lambda x: x**2 - 4

    result = solve_bisection(
        function,
        1,
        2
    )

    assert result == pytest.approx(2)


def test_solve_bisection_root_at_midpoint():

    function = lambda x: x**2 - 4

    result = solve_bisection(
        function,
        0,
        4
    )

    assert result == pytest.approx(2)


def test_solve_bisection_max_iterations():

    function = lambda x: x**2 - 2

    with pytest.raises(RuntimeError):
        solve_bisection(
            function,
            1,
            2,
            tolerance=1e-15,
            max_iterations=1
        )

# ============================================================
# LINEAR INTERPOLATION
# ============================================================

from pyadinic.numerical_methods import linear_interpolation


def test_linear_interpolation_basic():

    result = linear_interpolation(
        15,
        10,
        20,
        100,
        200
    )

    assert result == pytest.approx(150)


def test_linear_interpolation_lower_bound():

    result = linear_interpolation(
        10,
        10,
        20,
        100,
        200
    )

    assert result == pytest.approx(100)


def test_linear_interpolation_upper_bound():

    result = linear_interpolation(
        20,
        10,
        20,
        100,
        200
    )

    assert result == pytest.approx(200)


@pytest.mark.parametrize(
    "x, x0, x1, y0, y1, expected",
    [
        (12.5, 10, 20, 100, 200, 125),
        (15, 10, 20, 100, 200, 150),
        (17.5, 10, 20, 100, 200, 175),
        (5, 0, 10, -20, 20, 0),
        (2, 1, 5, 10, 30, 15),
    ]
)
def test_linear_interpolation_multiple_cases(
    x,
    x0,
    x1,
    y0,
    y1,
    expected
):

    result = linear_interpolation(
        x,
        x0,
        x1,
        y0,
        y1
    )

    assert result == pytest.approx(expected)


@pytest.mark.parametrize(
    "x, x0, x1",
    [
        (5, 10, 20),
        (25, 10, 20),
        (-1, 0, 10),
        (11, 0, 10),
    ]
)
def test_linear_interpolation_out_of_range(x, x0, x1):

    with pytest.raises(ValueError):
        linear_interpolation(
            x,
            x0,
            x1,
            100,
            200
        )


def test_linear_interpolation_zero_interval():

    with pytest.raises(ValueError):
        linear_interpolation(
            10,
            10,
            10,
            100,
            200
        )


@pytest.mark.parametrize(
    "x, x0, x1, y0, y1",
    [
        ("15", 10, 20, 100, 200),
        (15, "10", 20, 100, 200),
        (15, 10, "20", 100, 200),
        (15, 10, 20, "100", 200),
        (15, 10, 20, 100, "200"),
        (None, 10, 20, 100, 200),
        (15, None, 20, 100, 200),
        (15, 10, None, 100, 200),
        (15, 10, 20, None, 200),
        (15, 10, 20, 100, None),
    ]
)
def test_linear_interpolation_invalid_arguments(
    x,
    x0,
    x1,
    y0,
    y1
):

    with pytest.raises(TypeError):
        linear_interpolation(
            x,
            x0,
            x1,
            y0,
            y1
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_linear_interpolation_boolean_arguments(value):

    with pytest.raises(TypeError):
        linear_interpolation(
            value,
            10,
            20,
            100,
            200
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_linear_interpolation_boolean_x0(value):

    with pytest.raises(TypeError):
        linear_interpolation(
            15,
            value,
            20,
            100,
            200
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_linear_interpolation_boolean_x1(value):

    with pytest.raises(TypeError):
        linear_interpolation(
            15,
            10,
            value,
            100,
            200
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_linear_interpolation_boolean_y0(value):

    with pytest.raises(TypeError):
        linear_interpolation(
            15,
            10,
            20,
            value,
            200
        )


@pytest.mark.parametrize(
    "value",
    [
        True,
        False,
    ]
)
def test_linear_interpolation_boolean_y1(value):

    with pytest.raises(TypeError):
        linear_interpolation(
            15,
            10,
            20,
            100,
            value
        )

from pyadinic.numerical_methods import lagrange_interpolation
@pytest.mark.parametrize(
    "x, x_values, y_values, expected",
    [
        (5, [0, 10], [0, 20], 10),
        (1.5, [0, 1, 2], [0, 1, 4], 2.25),
        (2.5, [0, 1, 2, 3, 4], [0, 1, 4, 9, 16], 6.25),
    ],
)
def test_lagrange_interpolation(x, x_values, y_values, expected):

    result = lagrange_interpolation(
        x,
        x_values,
        y_values,
    )

    assert result == pytest.approx(expected)

@pytest.mark.parametrize(
    "x",
    [
        "1",
        None,
        ['invalid'],
    ],
)
def test_lagrange_invalid_x(x):

    with pytest.raises(TypeError):
        lagrange_interpolation(
            x,
            [0, 1],
            [0, 1],
        )


@pytest.mark.parametrize(
    "x_values",
    [
        ["0", 1],
        [0, None],
        [0, [1]],
    ],
)
def test_lagrange_invalid_x_values(x_values):

    with pytest.raises(TypeError):
        lagrange_interpolation(
            0.5,
            x_values,
            [0, 1],
        )


@pytest.mark.parametrize(
    "y_values",
    [
        ["0", 1],
        [0, None],
        [0, [1]],
    ],
)
def test_lagrange_invalid_y_values(y_values):

    with pytest.raises(TypeError):
        lagrange_interpolation(
            0.5,
            [0, 1],
            y_values,
        )


@pytest.mark.parametrize(
    "x, x_values, y_values",
    [
        (True, [0, 1], [0, 1]),
        (0.5, [False, 1], [0, 1]),
        (0.5, [0, 1], [False, 1]),
    ],
)
def test_lagrange_booleans(x, x_values, y_values):

    with pytest.raises(TypeError):
        lagrange_interpolation(
            x,
            x_values,
            y_values,
        )


@pytest.mark.parametrize(
    "x_values, y_values",
    [
        ([0, 1], [0]),
        ([0], [0, 1]),
        ([0, 1, 2], [0, 1]),
    ],
)
def test_lagrange_different_lengths(x_values, y_values):

    with pytest.raises(ValueError):
        lagrange_interpolation(
            0.5,
            x_values,
            y_values,
        )


@pytest.mark.parametrize(
    "x_values, y_values",
    [
        ([0], [0]),
        ([], []),
    ],
)
def test_lagrange_insufficient_points(x_values, y_values):

    with pytest.raises(ValueError):
        lagrange_interpolation(
            0.5,
            x_values,
            y_values,
        )


@pytest.mark.parametrize(
    "x_values",
    [
        [0, 1, 1],
        [0, 0],
        [1, 2, 3, 3],
    ],
)
def test_lagrange_repeated_x_values(x_values):

    with pytest.raises(ValueError):
        lagrange_interpolation(
            0.5,
            x_values,
            [0] * len(x_values),
        )


from pyadinic.numerical_methods.interpolation import (
    cubic_spline_interpolation,
)


@pytest.mark.parametrize(
    "x, x_values, y_values, expected",
    [
        (0, [0, 1, 2, 3], [0, 1, 4, 9], 0),
        (1, [0, 1, 2, 3], [0, 1, 4, 9], 1),
        (2, [0, 1, 2, 3], [0, 1, 4, 9], 4),
        (3, [0, 1, 2, 3], [0, 1, 4, 9], 9),
    ],
)
def test_cubic_spline_interpolation(
    x,
    x_values,
    y_values,
    expected,
):

    result = cubic_spline_interpolation(
        x,
        x_values,
        y_values,
    )

    assert result == pytest.approx(expected)


def test_cubic_spline_interior_point():

    result = cubic_spline_interpolation(
        1.5,
        [0, 1, 2, 3],
        [0, 1, 4, 9],
    )

    assert result == pytest.approx(2.2, abs=0.1)


def test_cubic_spline_endpoints():

    x_values = [0, 1, 2, 3]
    y_values = [2, 5, 3, 8]

    assert cubic_spline_interpolation(
        0,
        x_values,
        y_values,
    ) == pytest.approx(2)

    assert cubic_spline_interpolation(
        3,
        x_values,
        y_values,
    ) == pytest.approx(8)


@pytest.mark.parametrize(
    "x",
    [
        "1",
        None,
        ['invalid'],
    ],
)
def test_cubic_spline_invalid_x(x):

    with pytest.raises(TypeError):

        cubic_spline_interpolation(
            x,
            [0, 1, 2],
            [0, 1, 4],
        )


@pytest.mark.parametrize(
    "x_values",
    [
        ["0", 1, 2],
        [0, None, 2],
        [0, 1, [2]],
    ],
)
def test_cubic_spline_invalid_x_values(x_values):

    with pytest.raises(TypeError):

        cubic_spline_interpolation(
            0.5,
            x_values,
            [0, 1, 4],
        )


@pytest.mark.parametrize(
    "y_values",
    [
        ["0", 1, 4],
        [0, None, 4],
        [0, 1, [4]],
    ],
)
def test_cubic_spline_invalid_y_values(y_values):

    with pytest.raises(TypeError):

        cubic_spline_interpolation(
            0.5,
            [0, 1, 2],
            y_values,
        )


@pytest.mark.parametrize(
    "x, x_values, y_values",
    [
        (True, [0, 1, 2], [0, 1, 4]),
        (0.5, [False, 1, 2], [0, 1, 4]),
        (0.5, [0, 1, 2], [False, 1, 4]),
    ],
)
def test_cubic_spline_booleans(
    x,
    x_values,
    y_values,
):

    with pytest.raises(TypeError):

        cubic_spline_interpolation(
            x,
            x_values,
            y_values,
        )


@pytest.mark.parametrize(
    "x_values, y_values",
    [
        ([0, 1, 2], [0, 1]),
        ([0, 1], [0, 1, 4]),
        ([0, 1, 2, 3], [0, 1, 4]),
    ],
)
def test_cubic_spline_different_lengths(
    x_values,
    y_values,
):

    with pytest.raises(ValueError):

        cubic_spline_interpolation(
            0.5,
            x_values,
            y_values,
        )


@pytest.mark.parametrize(
    "x_values, y_values",
    [
        ([0, 1], [0, 1]),
        ([0], [0]),
        ([], []),
    ],
)
def test_cubic_spline_insufficient_points(
    x_values,
    y_values,
):

    with pytest.raises(ValueError):

        cubic_spline_interpolation(
            0.5,
            x_values,
            y_values,
        )


@pytest.mark.parametrize(
    "x_values",
    [
        [0, 1, 1],
        [0, 0, 2],
        [0, 1, 2, 2],
    ],
)
def test_cubic_spline_repeated_x_values(x_values):

    with pytest.raises(ValueError):

        cubic_spline_interpolation(
            0.5,
            x_values,
            [0] * len(x_values),
        )


@pytest.mark.parametrize(
    "x",
    [
        -1,
        4,
    ],
)
def test_cubic_spline_out_of_range(x):

    with pytest.raises(ValueError):

        cubic_spline_interpolation(
            x,
            [0, 1, 2, 3],
            [0, 1, 4, 9],
        )

from pyadinic.numerical_methods.interpolation import (
    pchip_interpolation
)
 
def test_pchip_interpolation():

    x_values = [0, 1, 2, 3]
    y_values = [0, 1, 4, 9]

    assert pchip_interpolation(
        0, x_values, y_values
    ) == pytest.approx(0)

    assert pchip_interpolation(
        1, x_values, y_values
    ) == pytest.approx(1)

    assert pchip_interpolation(
        2, x_values, y_values
    ) == pytest.approx(4)

    assert pchip_interpolation(
        3, x_values, y_values
    ) == pytest.approx(9)

    assert pchip_interpolation(
        1.5, x_values, y_values
    ) == pytest.approx(2.21875)


def test_pchip_interpolation_invalid_x():

    with pytest.raises(TypeError):
        pchip_interpolation(
            "1.5",
            [0, 1, 2],
            [0, 1, 4]
        )

    with pytest.raises(TypeError):
        pchip_interpolation(
            True,
            [0, 1, 2],
            [0, 1, 4]
        )


def test_pchip_interpolation_invalid_data():

    with pytest.raises(ValueError):
        pchip_interpolation(
            1,
            [0, 1],
            [0, 1]
        )

    with pytest.raises(ValueError):
        pchip_interpolation(
            1,
            [0, 1, 2],
            [0, 1]
        )

    with pytest.raises(ValueError):
        pchip_interpolation(
            1,
            [0, 1, 1],
            [0, 1, 4]
        )


def test_pchip_interpolation_out_of_range():

    with pytest.raises(ValueError):
        pchip_interpolation(
            -1,
            [0, 1, 2],
            [0, 1, 4]
        )

    with pytest.raises(ValueError):
        pchip_interpolation(
            3,
            [0, 1, 2],
            [0, 1, 4]
        )


def test_pchip_interpolation_non_numeric_data():

    with pytest.raises(TypeError):
        pchip_interpolation(
            1,
            [0, "1", 2],
            [0, 1, 4]
        )

    with pytest.raises(TypeError):
        pchip_interpolation(
            1,
            [0, 1, 2],
            [0, "1", 4]
        )

from pyadinic.numerical_methods.integration import legendre_roots, legendre_weights, integrate_gauss_legendre

def test_legendre_roots_three_points():
    roots = legendre_roots(3)

    assert roots[0] == pytest.approx(-0.7745966692)
    assert roots[1] == pytest.approx(0.0)
    assert roots[2] == pytest.approx(0.7745966692)


def test_legendre_weights():

    weights_2 = legendre_weights(2)
    expected_2 = [1.0, 1.0]

    weights_3 = legendre_weights(3)
    expected_3 = [5/9, 8/9, 5/9]

    weights_4 = legendre_weights(4)
    expected_4 = [
        0.3478548451,
        0.6521451549,
        0.6521451549,
        0.3478548451
    ]

    for weight, expected in zip(weights_2, expected_2):
        assert weight == pytest.approx(expected)

    for weight, expected in zip(weights_3, expected_3):
        assert weight == pytest.approx(expected)

    for weight, expected in zip(weights_4, expected_4):
        assert weight == pytest.approx(expected)


def test_integrate_gauss_legendre():

    result = integrate_gauss_legendre(
        lambda x: x**2,
        0,
        2,
        2
    )

    assert result == pytest.approx(8/3)


def test_cubic_spline_unsorted_x():
    with pytest.raises(ValueError):
        cubic_spline_interpolation(1.5, [0, 3, 1], [0, 9, 1])


def test_pchip_unsorted_x():
    with pytest.raises(ValueError):
        pchip_interpolation(1.5, [0, 3, 1], [0, 9, 1])


def test_integrate_gauss_legendre_validations():
    with pytest.raises(TypeError):
        integrate_gauss_legendre("not_callable", 0, 1, 2)
    with pytest.raises(TypeError):
        integrate_gauss_legendre(lambda x: x, "0", 1, 2)
    with pytest.raises(TypeError):
        integrate_gauss_legendre(lambda x: x, 0, 1, 2.5)
    with pytest.raises(ValueError):
        integrate_gauss_legendre(lambda x: x, 0, 1, 0)
