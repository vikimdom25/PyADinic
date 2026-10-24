
def integrate_trapezoidal(function, a, b, n):

    if not callable(function):
        raise TypeError("Function must be callable")

    mags = [a,b,n]

    for mag in mags:
        if not isinstance(mag, (int,float)) or isinstance(mag,bool):
            raise TypeError("One of the arguments required is not numeric")

    if (n <= 0) or (isinstance(n,float)):
        raise ValueError("Number of trapezoids must be a positive integer")

    gaps = (b - a)/n
    resultado = (function(a) + function(b))/2

    for i in range(1,n):
        x_i = a + (i*gaps)
        resultado += function(x_i)

    resultado *= gaps
    return resultado

def integrate_simpson(function, a, b, n):

    if not callable(function):
        raise TypeError("Function must be callable")

    mags = [a,b,n]

    for mag in mags:
        if not isinstance(mag, (int,float)) or isinstance(mag,bool):
            raise TypeError("One of the arguments required is not numeric")

    if (n <= 0) or (isinstance(n,float)) or (n % 2 != 0):
        raise ValueError("Number of intervals must be a positive integer and even")

    gaps = (b - a)/n
    resultado = (function(a) + function(b))

    for i in range(1,n):
        x_i = a + (i*gaps)

        if (i % 2 == 0):
            resultado += 2*function(x_i)
        else:
            resultado += 4*function(x_i)


    resultado *= gaps/3
    return resultado

def integrate_simpson_38(function, a, b, n):

    if not callable(function):
        raise TypeError("Function must be callable")
    
    mags = [a,b,n]
    
    for mag in mags:
        if not isinstance(mag, (int,float)) or isinstance(mag,bool):
            raise TypeError("One of the arguments required is not numeric")
    
    if (n <= 0) or (isinstance(n, float)) or (n % 3 != 0):
        raise ValueError(
            "Number of intervals must be a positive integer and a multiple of 3"
        )

    gaps = (b - a) / n
    resultado = function(a) + function(b)

    for i in range(1, n):
        x_i = a + i * gaps

        if i % 3 == 0:
            resultado += 2 * function(x_i)
        else:
            resultado += 3 * function(x_i)

    resultado *= 3 * gaps / 8
    return resultado    

def legendre_polynomial(n, x):

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")

    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError("x must be numeric")

    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0:
        return 1.0

    if n == 1:
        return float(x)

    p_previous = 1.0
    p_current = float(x)

    for order in range(1, n):
        p_next = (
            ((2 * order + 1) * x * p_current)
            - (order * p_previous)
        ) / (order + 1)

        p_previous = p_current
        p_current = p_next

    return p_current

def legendre_derivative(n, x):

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")

    if not isinstance(x, (int, float)) or isinstance(x, bool):
        raise TypeError("x must be numeric")

    if n < 0:
        raise ValueError("n must be non-negative")

    if n == 0:
        return 0.0

    if x == 1 or x == -1:
        raise ValueError("Derivative is undefined with this expression at x = ±1")

    p_n = legendre_polynomial(n, x)
    p_previous = legendre_polynomial(n - 1, x)

    return (n / (x**2 - 1)) * (x * p_n - p_previous)

def legendre_roots(n):

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be positive")

    roots = []

    from .solvers import solve_newton
    import math

    for i in range(1, n + 1):

        initial_guess = math.cos(
            math.pi * (i - 0.25) / (n + 0.5)
        )

        root = solve_newton(
            lambda x: legendre_polynomial(n, x),
            initial_guess,
            tolerance=1e-12,
            max_iterations=100
        )

        roots.append(root)

    return roots[::-1]

def legendre_weights(n):

    roots = legendre_roots(n)
    weights = []

    for root in roots:
        derivative = legendre_derivative(n, root)

        weight = 2 / ((1 - root**2) * derivative**2)

        weights.append(weight)

    return weights

def integrate_gauss_legendre(function, a, b, n):
    if not callable(function):
        raise TypeError("Function must be callable")

    for mag in (a, b):
        if not isinstance(mag, (int, float)) or isinstance(mag, bool):
            raise TypeError("Integration limits must be numeric")

    if not isinstance(n, int) or isinstance(n, bool):
        raise TypeError("n must be an integer")

    if n <= 0:
        raise ValueError("n must be positive")

    roots = legendre_roots(n)
    weights = legendre_weights(n)

    resultado = 0

    for root, weight in zip(roots, weights):
        x = ((b - a) / 2) * root + ((a + b) / 2)
        resultado += weight * function(x)

    resultado *= (b - a) / 2
    return resultado