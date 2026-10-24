import numpy as np


def euler(function, t_span, y0, h):

    if not callable(function):
        raise TypeError("Function must be callable")

    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise TypeError("t_span must contain t0 and tf")

    t0, tf = t_span

    for value in (t0, tf, h):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("t0, tf and h must be numeric")

    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    if h <= 0:
        raise ValueError("h must be greater than zero")

    y = np.asarray(y0, dtype=float)

    t = t0

    t_values = [t]
    y_values = [y.copy()]

    tol = min(h * 1e-5, 1e-12 * max(1.0, abs(tf), abs(t0)))

    while t < tf - tol:

        if tf - t <= h + tol:
            step = tf - t
            next_t = tf
        else:
            step = h
            next_t = t + h

        dydt = np.asarray(function(t, y), dtype=float)

        if dydt.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y = y + (step * dydt)
        t = next_t

        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)


def heun(function, t_span, y0, h):

    if not callable(function):
        raise TypeError("Function must be callable")

    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise TypeError("t_span must contain t0 and tf")

    t0, tf = t_span

    for value in (t0, tf, h):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("t0, tf and h must be numeric")

    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    if h <= 0:
        raise ValueError("h must be greater than zero")

    y = np.asarray(y0, dtype=float)

    t = t0

    t_values = [t]
    y_values = [y.copy()]

    tol = min(h * 1e-5, 1e-12 * max(1.0, abs(tf), abs(t0)))

    while t < tf - tol:

        if tf - t <= h + tol:
            step = tf - t
            next_t = tf
        else:
            step = h
            next_t = t + h

        k1 = np.asarray(function(t, y), dtype=float)

        if k1.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y_star = y + (step * k1)

        k2 = np.asarray(function(t + step, y_star), dtype=float)

        if k2.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y = y + ((step / 2) * (k1 + k2))
        t = next_t

        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)


def rk4(function, t_span, y0, h):

    if not callable(function):
        raise TypeError("Function must be callable")

    if not isinstance(t_span, (tuple, list)) or len(t_span) != 2:
        raise TypeError("t_span must contain t0 and tf")

    t0, tf = t_span

    for value in (t0, tf, h):
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise TypeError("t0, tf and h must be numeric")

    if tf <= t0:
        raise ValueError("tf must be greater than t0")

    if h <= 0:
        raise ValueError("h must be greater than zero")

    y = np.asarray(y0, dtype=float)

    t = t0

    t_values = [t]
    y_values = [y.copy()]

    tol = min(h * 1e-5, 1e-12 * max(1.0, abs(tf), abs(t0)))

    while t < tf - tol:

        if tf - t <= h + tol:
            step = tf - t
            next_t = tf
        else:
            step = h
            next_t = t + h

        k1 = np.asarray(
            function(t, y),
            dtype=float
        )

        if k1.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y_star = y + (step / 2) * k1

        k2 = np.asarray(
            function(t + step / 2, y_star),
            dtype=float
        )

        if k2.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y_star = y + (step / 2) * k2

        k3 = np.asarray(
            function(t + step / 2, y_star),
            dtype=float
        )

        if k3.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y_star = y + step * k3

        k4 = np.asarray(
            function(t + step, y_star),
            dtype=float
        )

        if k4.shape != y.shape:
            raise ValueError(
                "Function output must have the same shape as y0"
            )

        y = y + (step / 6) * (
            k1 + 2 * k2 + 2 * k3 + k4
        )

        t = next_t

        t_values.append(t)
        y_values.append(y.copy())

    return np.array(t_values), np.array(y_values)
