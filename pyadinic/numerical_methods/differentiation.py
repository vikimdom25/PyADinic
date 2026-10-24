def numerical_derivative(function, x, h=1e-6):

    if not callable(function):
        raise TypeError("Function must be callable")

    numeric_args = [x, h]

    for arg in numeric_args:
        if not isinstance(arg, (int, float)):
            raise TypeError("One of the arguments required is not numeric")
        if isinstance(arg, bool):
            raise TypeError("No booleans allowed")

    if h <= 0:
        raise ValueError("Step size h must be positive")

    return (function(x + h) - function(x - h)) / (2 * h)