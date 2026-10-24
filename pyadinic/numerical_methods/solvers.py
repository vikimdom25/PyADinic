def solve_brent(function, lb, up, tolerance=1e-8, max_iterations=100):

    if not callable(function):
        raise TypeError("Function must be callable")

    numeric_mags = [lb, up, tolerance, max_iterations]

    for mag in numeric_mags:
        if not isinstance(mag, (int, float)):
            raise TypeError("One of parameters is not numeric")
        if isinstance(mag, bool):
            raise TypeError("Not booleans params allowed")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive and not zero")

    if max_iterations <= 0:
        raise ValueError("Iterations must be positive and not zero")

    # Evaluamos la función solamente una vez en cada extremo
    f_lb = function(lb)
    f_up = function(up)

    if f_lb == 0:
        return lb

    if f_up == 0:
        return up

    if f_lb * f_up > 0:
        raise ValueError(
            "Unable to secure a root in the range described"
        )

    # Inicialización de Brent
    a = lb
    b = up
    c = up

    fa = f_lb
    fb = f_up
    fc = f_up

    # Memoria de pasos anteriores
    d = b - a
    e = d

    for _ in range(max_iterations):

        # b debe ser la mejor aproximación
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        # b y c deben mantener el bracket
        if fb * fc > 0:
            c = a
            fc = fa
            d = b - a
            e = d

        # b debe seguir siendo la mejor aproximación
        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        # Convergencia
        midpoint = 0.5 * (c - b)

        if abs(midpoint) <= tolerance or fb == 0:
            return b

        # --------------------------------------------------
        # Decidir entre interpolación y bisección
        # --------------------------------------------------

        if abs(e) >= tolerance and abs(fa) > abs(fb):

            # --------------------------------------------------
            # Interpolación
            # --------------------------------------------------

            s = fb / fa

            if a == c:
                # Secante
                p = 2 * midpoint * s
                q = 1 - s

            else:
                # IQI
                q_ratio = fa / fc
                r = fb / fc

                p = s * (
                    2 * midpoint * q_ratio * (q_ratio - r)
                    - (b - a) * (r - 1)
                )

                q = (
                    (q_ratio - 1)
                    * (r - 1)
                    * (s - 1)
                )

            # Normalizamos el signo
            if p > 0:
                q = -q
            else:
                p = -p

            # --------------------------------------------------
            # ¿Aceptamos interpolación?
            # --------------------------------------------------

            if (
                2 * p < min(
                    3 * midpoint * q - abs(tolerance * q),
                    abs(e * q)
                )
            ):
                e = d
                d = p / q

            else:
                # Bisección
                d = midpoint
                e = midpoint

        else:
            # --------------------------------------------------
            # Bisección
            # --------------------------------------------------

            d = midpoint
            e = midpoint

        # --------------------------------------------------
        # Nuevo punto
        # --------------------------------------------------

        a = b
        fa = fb

        if abs(d) > tolerance:
            b = b + d
        elif midpoint > 0:
            b = b + tolerance
        else:
            b = b - tolerance

        fb = function(b)

    raise RuntimeError("Maximum number of iterations exceeded")

def solve_newton(function, x0, tolerance=1e-8, max_iterations=100):

    if not callable(function):
        raise TypeError("Function must be callable")

    mags = [x0, tolerance, max_iterations]

    for mag in mags:
        if not isinstance(mag, (int, float)):
            raise TypeError("One of the arguments required is not numeric")
        if isinstance(mag, bool):
            raise TypeError("No booleans allowed")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")

    x = x0
    from .differentiation import numerical_derivative

    for _ in range(max_iterations):

        fx = function(x)

        derivative = numerical_derivative(function, x)

        if derivative == 0:
            raise ValueError("Zero derivative encountered")

        x_new = x - fx / derivative

        if abs(x_new - x) <= tolerance:
            return x_new

        x = x_new

    raise RuntimeError("Maximum number of iterations exceeded")

def solve_secant(function, x0, x1, tolerance=1e-8, max_iterations=100):
    if not callable(function):
        raise TypeError("Function must be callable")

    mags = [x0, x1, tolerance, max_iterations]

    for mag in mags:
        if not isinstance(mag, (int, float)):
            raise TypeError("One of the arguments required is not numeric")
        if isinstance(mag, bool):
            raise TypeError("No booleans allowed")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")

    fx_actual = function(x1)
    fx_anterior = function(x0)

    for _ in range(max_iterations):

        if (fx_actual - fx_anterior) == 0:
            raise ValueError("Zero secant denominator")
        
        x_tentativo = x1 - (fx_actual*((x1 -  x0)/(fx_actual - fx_anterior)))

        if abs(x_tentativo - x1) <= tolerance:
            return x_tentativo

        x0 = x1
        x1 = x_tentativo

        fx_anterior = fx_actual
        fx_actual = function(x1)

    raise RuntimeError("Maximum number of iterations exceeded")

def solve_bisection(function, lb, up, tolerance=1e-8, max_iterations=100):

    if not callable(function):
        raise TypeError("Function must be callable")

    mags = [lb, up, tolerance, max_iterations]

    for mag in mags:
        if not isinstance(mag, (int, float)):
            raise TypeError("One of the arguments required is not numeric")
        if isinstance(mag, bool):
            raise TypeError("No booleans allowed")

    if tolerance <= 0:
        raise ValueError("Tolerance must be positive")

    if max_iterations <= 0:
        raise ValueError("Maximum iterations must be positive")

    f_lb = function(lb)
    f_up = function(up)

    if f_lb * f_up > 0:
        raise ValueError("Unable to secure a root in the range described")

    for _ in range(max_iterations):

        midpoint = (lb + up)/2
        f_mid = function(midpoint)

        if f_mid == 0:
            return midpoint

        if f_lb * f_mid > 0:
            lb = midpoint
            f_lb = f_mid

        if f_up * f_mid > 0:
            up = midpoint
            f_up = f_mid

        if abs(up - lb) <= tolerance:
            return midpoint

    raise RuntimeError("Maximum number of iterations exceeded")

def solve_tridiagonal(lower, diagonal, upper, right_side):

    n = len(diagonal)

    if len(lower) != n - 1:
        raise ValueError("Invalid lower diagonal length")

    if len(upper) != n - 1:
        raise ValueError("Invalid upper diagonal length")

    if len(right_side) != n:
        raise ValueError("Invalid right side length")

    diagonal = [float(value) for value in diagonal]
    lower = [float(value) for value in lower]
    upper = [float(value) for value in upper]
    right_side = [float(value) for value in right_side]

    for i in range(1, n):

        factor = lower[i - 1] / diagonal[i - 1]

        diagonal[i] -= factor * upper[i - 1]
        right_side[i] -= factor * right_side[i - 1]

    solution = [0.0] * n

    solution[-1] = right_side[-1] / diagonal[-1]

    for i in range(n - 2, -1, -1):

        solution[i] = (
            right_side[i]
            - upper[i] * solution[i + 1]
        ) / diagonal[i]

    return solution
    