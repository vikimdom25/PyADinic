
import numpy as np
import pytest

from pyadinic.numerical_methods.ode import euler, heun, rk4


def test_euler_scalar():
    t, y = euler(
        lambda t, y: y,
        (0, 1),
        1.0,
        0.1
    )

    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(2.59374246)


def test_euler_system():
    def oscillator(t, y):
        x, v = y
        return np.array([v, -x])

    t, y = euler(
        oscillator,
        (0, 1),
        np.array([1.0, 0.0]),
        0.1
    )

    assert t[-1] == pytest.approx(1.0)

    assert y[-1, 0] == pytest.approx(
        np.cos(1),
        abs=0.1
    )

    assert y[-1, 1] == pytest.approx(
        -np.sin(1),
        abs=0.1
    )


def test_euler_partial_step():
    t, y = euler(
        lambda t, y: y,
        (0, 1),
        1.0,
        0.3
    )

    assert t[-2] == pytest.approx(0.9)
    assert t[-1] == pytest.approx(1.0)


def test_euler_convergence():
    exact = np.e

    _, y_h1 = euler(
        lambda t, y: y,
        (0, 1),
        1.0,
        0.1
    )

    _, y_h2 = euler(
        lambda t, y: y,
        (0, 1),
        1.0,
        0.05
    )

    error_h1 = abs(y_h1[-1] - exact)
    error_h2 = abs(y_h2[-1] - exact)

    assert error_h2 < error_h1
    assert error_h2 / error_h1 == pytest.approx(0.5, rel=0.1)


def test_euler_invalid_function():
    with pytest.raises(TypeError):
        euler(
            5,
            (0, 1),
            1.0,
            0.1
        )


def test_euler_invalid_span():
    with pytest.raises(ValueError):
        euler(
            lambda t, y: y,
            (1, 0),
            1.0,
            0.1
        )


def test_euler_invalid_step():
    with pytest.raises(ValueError):
        euler(
            lambda t, y: y,
            (0, 1),
            1.0,
            0
        )


def test_euler_incompatible_state():
    def function(t, y):
        return np.array([y[0], y[1], 0.0])

    with pytest.raises(ValueError):
        euler(
            function,
            (0, 1),
            np.array([1.0, 2.0]),
            0.1
        )

def test_heun_scalar():
    def function(t, y):
        return y

    t, y = heun(
        function,
        (0, 1),
        1.0,
        0.1
    )

    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(np.exp(1), abs=5e-3)


def test_heun_system():
    def function(t, y):
        return np.array([
            y[1],
            -y[0]
        ])

    t, y = heun(
        function,
        (0, 1),
        np.array([1.0, 0.0]),
        0.1
    )

    expected = np.array([
        np.cos(1),
        -np.sin(1)
    ])

    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(expected, abs=2e-3)


def test_heun_partial_final_step():
    def function(t, y):
        return np.ones_like(y)

    t, y = heun(
        function,
        (0, 1),
        0.0,
        0.3
    )

    assert t[-2] == pytest.approx(0.9)
    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(1.0)


def test_heun_convergence_order():
    def function(t, y):
        return y

    _, y_h1 = heun(
        function,
        (0, 1),
        1.0,
        0.1
    )

    _, y_h2 = heun(
        function,
        (0, 1),
        1.0,
        0.05
    )

    exact = np.exp(1)

    error_h1 = abs(y_h1[-1] - exact)
    error_h2 = abs(y_h2[-1] - exact)

    ratio = error_h2 / error_h1

    assert ratio == pytest.approx(0.25, rel=0.1)


def test_heun_invalid_function():
    with pytest.raises(TypeError):
        heun(
            123,
            (0, 1),
            1.0,
            0.1
        )


def test_heun_invalid_t_span():
    with pytest.raises(TypeError):
        heun(
            lambda t, y: y,
            0,
            1.0,
            0.1
        )

    with pytest.raises(TypeError):
        heun(
            lambda t, y: y,
            (0, 1, 2),
            1.0,
            0.1
        )


def test_heun_invalid_interval():
    with pytest.raises(ValueError):
        heun(
            lambda t, y: y,
            (1, 0),
            1.0,
            0.1
        )


def test_heun_invalid_step():
    with pytest.raises(ValueError):
        heun(
            lambda t, y: y,
            (0, 1),
            1.0,
            0
        )

    with pytest.raises(ValueError):
        heun(
            lambda t, y: y,
            (0, 1),
            1.0,
            -0.1
        )


def test_heun_incompatible_state():
    def function(t, y):
        return np.array([y[0], y[1], 0.0])

    with pytest.raises(ValueError):
        heun(
            function,
            (0, 1),
            np.array([1.0, 0.0]),
            0.1
        )


def test_rk4_scalar():
    def function(t, y):
        return y

    t, y = rk4(
        function,
        (0, 1),
        1.0,
        0.1
    )

    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(np.exp(1), abs=1e-5)


def test_rk4_system():
    def function(t, y):
        return np.array([
            y[1],
            -y[0]
        ])

    t, y = rk4(
        function,
        (0, 1),
        np.array([1.0, 0.0]),
        0.1
    )

    expected = np.array([
        np.cos(1),
        -np.sin(1)
    ])

    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(expected, abs=1e-5)


def test_rk4_partial_final_step():
    def function(t, y):
        return np.ones_like(y)

    t, y = rk4(
        function,
        (0, 1),
        0.0,
        0.3
    )

    assert t[-2] == pytest.approx(0.9)
    assert t[-1] == pytest.approx(1.0)
    assert y[-1] == pytest.approx(1.0)


def test_rk4_convergence_order():
    def function(t, y):
        return y

    _, y_h1 = rk4(
        function,
        (0, 1),
        1.0,
        0.1
    )

    _, y_h2 = rk4(
        function,
        (0, 1),
        1.0,
        0.05
    )

    exact = np.exp(1)

    error_h1 = abs(y_h1[-1] - exact)
    error_h2 = abs(y_h2[-1] - exact)

    ratio = error_h2 / error_h1

    assert ratio == pytest.approx(1 / 16, rel=0.1)


def test_rk4_invalid_function():
    with pytest.raises(TypeError):
        rk4(
            123,
            (0, 1),
            1.0,
            0.1
        )


def test_rk4_invalid_t_span():
    with pytest.raises(TypeError):
        rk4(
            lambda t, y: y,
            0,
            1.0,
            0.1
        )

    with pytest.raises(TypeError):
        rk4(
            lambda t, y: y,
            (0, 1, 2),
            1.0,
            0.1
        )


def test_rk4_invalid_interval():
    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            (1, 0),
            1.0,
            0.1
        )


def test_rk4_invalid_step():
    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            (0, 1),
            1.0,
            0
        )

    with pytest.raises(ValueError):
        rk4(
            lambda t, y: y,
            (0, 1),
            1.0,
            -0.1
        )


def test_rk4_incompatible_state():
    def function(t, y):
        return np.array([
            y[0],
            y[1],
            0.0
        ])

    with pytest.raises(ValueError):
        rk4(
            function,
            (0, 1),
            np.array([1.0, 0.0]),
            0.1
        )

