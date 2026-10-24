import math 
import numpy as np
from pyadinic.core.dataclass import Airfoil

def parse_naca4(chain: str):

    chain = chain

    if not isinstance(chain, str):
        raise TypeError("string of numbers as argument needed")

    if len(chain) != 4:
        raise TypeError("No 4 digits")

    try:
        for digit in chain:
            float(digit)
    except ValueError:
        raise TypeError("String not convertible in a numeric value")
        

    first = chain[0]
    second = chain[1]
    last = chain[2:]

    m = float(first)/100
    p = float(second)/10
    t = float(last)/100


    if not (0 <= m <= 0.09):
        raise ValueError("Max camber out of range")
    if m == 0 and p == 0:
        pass
    elif not (0.1 <= p <= 0.9):
        raise ValueError("Camber max position out of range")
    if not (0 <= t <= 0.99):
        raise ValueError("Tickness out of range")
    
    return m, p, t


def thickness_distribution(x,t):
    if not (0 <= x <= 1):
        raise ValueError("Normalized Chord, x out of range")

    y_t = (5*t) * ((0.2969 * math.sqrt(x)) - (0.1260 * x) - (0.3516 * (x**2)) + (0.2843 * (x**3)) - (0.1015 *  (x**4))) 
    
    return y_t 


def camber_line(x, m, p):

    if not (0 <= x <= 1):
        raise ValueError("Normalized chord, x out of range")

    if m == 0 or p == 0:
        return 0.0
    
    if (0 <= x <= p):

        y_e = ((m/(p ** 2)) * ((2*p*x) - (x**2)))

        return y_e
    
    if (p < x <= 1):

        y_e = ((m/((1 - p) ** 2)) * ((1 - (2*p)) + (2*p*x) - (x**2)))

        return y_e


def camber_line_slope(x, m, p):

    if not (0 <= x <= 1):
        raise ValueError("Normalized chord, x out of range")

    if m == 0 or p == 0:
        return 0.0

    if 0 <= x <= p:

        slope = (2*m/(p**2)) * (p-x)

        return slope

    if p < x <= 1:

        slope = (2*m/((1-p)**2)) * (p-x)

        return slope


def surface_coordinates(x, m, p, t):

    y_c = camber_line(x, m, p)
    y_t = thickness_distribution(x, t)
    slope = camber_line_slope(x, m, p)

    theta = math.atan(slope)

    x_upper = x - y_t * math.sin(theta)
    y_upper = y_c + y_t * math.cos(theta)

    x_lower = x + y_t * math.sin(theta)
    y_lower = y_c - y_t * math.cos(theta)

    return x_upper, y_upper, x_lower, y_lower


def spacing(n_points: int, spacing_type="cosine"):

    if n_points < 2:
        raise ValueError("n_points must be at least 2")
        
    if spacing_type == "regular":
        return np.linspace(0, 1, n_points)

    if spacing_type  == "cosine":
        cosine_space = np.zeros(n_points)

        for i in range(n_points):

            point = (1 - math.cos(
                (i * math.pi) / (n_points - 1)
            )) / 2

            cosine_space[i] = point

        return cosine_space
    else:
        raise ValueError("Unknown spacing type")

    
def naca4(code, chord=1.0, n_points=100, spacing_type="cosine"):

    m, p, t = parse_naca4(code)

    x = spacing(n_points, spacing_type=spacing_type)

    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []

    if chord <= 0:
        raise ValueError("Chord must be positive")

    for x_value in x:
        xu, yu, xl, yl = surface_coordinates(
            x_value, m, p, t
        )

        x_upper.append(xu)
        y_upper.append(yu)
        x_lower.append(xl)
        y_lower.append(yl)

    return Airfoil(
        name=f"NACA {code}",
        x_upper=np.array(x_upper) * chord,
        y_upper=np.array(y_upper) * chord,
        x_lower=np.array(x_lower) * chord,
        y_lower=np.array(y_lower) * chord,
    )


#NACA DE 5 DIGITOS

def parse_naca5(chain: str):

    if not isinstance(chain, str):
        raise TypeError("string of numbers as argument needed")

    if len(chain) != 5:
        raise TypeError("No 5 digits")

    try:
        for digit in chain:
            float(digit)
    except ValueError:
        raise TypeError("String not convertible in a numeric value")

    first = int(chain[0])
    second = int(chain[1])
    third = int(chain[2])
    last = int(chain[3:])

    if not (1 <= first <= 5):
        raise ValueError("Design lift coefficient digit out of range")

    if not (1 <= second <= 5):
        raise ValueError("Maximum camber position digit out of range")

    if third not in (0, 1):
        raise ValueError("Camber type digit must be 0 or 1")

    if not (0 <= last <= 99):
        raise ValueError("Thickness out of range")

    if third == 1 and second == 1:
        raise ValueError(
            "NACA 211 profiles are not supported"
        )

    design_lift_coefficient = 0.15 * first
    max_camber_position = 0.05 * second
    thickness = last / 100

    standard_parameters = {
        1: (0.0580, 361.400),
        2: (0.1260, 51.640),
        3: (0.2025, 15.957),
        4: (0.2900, 6.643),
        5: (0.3910, 3.230),
    }

    reflex_parameters = {
        2: (0.1300, 51.990, 0.000764),
        3: (0.2170, 15.793, 0.00677),
        4: (0.3180, 6.520, 0.0303),
        5: (0.4410, 3.191, 0.1355),
    }

    if third == 0:

        r, k1_reference = standard_parameters[second]

        k1 = k1_reference * (
            design_lift_coefficient / 0.3
        )

        k2_ratio = 0.0

    else:

        r, k1_reference, k2_ratio = reflex_parameters[second]

        k1 = k1_reference * (
            design_lift_coefficient / 0.3
        )

    return (
        design_lift_coefficient,
        max_camber_position,
        third,
        thickness,
        r,
        k1,
        k2_ratio,
    )

def camber_line_5(x,r,k1,reflex=False,k2_ratio=0.0,):

    if not (0 <= x <= 1):
        raise ValueError("Normalized Chord, x out of range")

    if not reflex:

        if 0 <= x <= r:

            y_c = (
                k1 / 6
            ) * (
                (x ** 3)
                - (3 * r * x ** 2)
                + (r ** 2 * (3 - r) * x)
            )

            return y_c

        y_c = (
            k1 * (r ** 3) / 6
        ) * (1 - x)

        return y_c

    if 0 <= x <= r:

        y_c = (
            k1 / 6
        ) * (
            (x - r) ** 3
            - k2_ratio * ((1 - r) ** 3) * x
            - (r ** 3) * x
            + r ** 3
        )

        return y_c

    y_c = (
        k1 / 6
    ) * (
        k2_ratio * ((x - r) ** 3)
        - k2_ratio * ((1 - r) ** 3) * x
        - (r ** 3) * x
        + r ** 3
    )

    return y_c

def camber_line_slope_5(x,r,k1,reflex=False,k2_ratio=0.0,):

    if not (0 <= x <= 1):
        raise ValueError("Normalized Chord, x out of range")

    if not reflex:

        if 0 <= x <= r:

            slope = (
                k1 / 6
            ) * (
                3 * (x ** 2)
                - 6 * r * x
                + (r ** 2 * (3 - r))
            )

            return slope

        slope = (
            -k1 * (r ** 3) / 6
        )

        return slope

    if 0 <= x <= r:

        slope = (
            k1 / 6
        ) * (
            3 * ((x - r) ** 2)
            - k2_ratio * ((1 - r) ** 3)
            - r ** 3
        )

        return slope

    slope = (
        k1 / 6
    ) * (
        3 * k2_ratio * ((x - r) ** 2)
        - k2_ratio * ((1 - r) ** 3)
        - r ** 3
    )

    return slope

def naca5(code, chord=1.0, n_points=100, spacing_type="cosine"):

    (
        design_lift_coefficient,
        max_camber_position,
        camber_type,
        t,
        r,
        k1,
        k2_ratio,
    ) = parse_naca5(code)

    x = spacing(n_points, spacing_type=spacing_type)

    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []

    reflex = camber_type == 1

    for x_value in x:

        y_c = camber_line_5(
            x_value,
            r,
            k1,
            reflex,
            k2_ratio,
        )

        y_t = thickness_distribution(
            x_value,
            t,
        )

        slope = camber_line_slope_5(
            x_value,
            r,
            k1,
            reflex,
            k2_ratio,
        )

        theta = math.atan(slope)

        xu = (
            x_value
            - y_t * math.sin(theta)
        )

        yu = (
            y_c
            + y_t * math.cos(theta)
        )

        xl = (
            x_value
            + y_t * math.sin(theta)
        )

        yl = (
            y_c
            - y_t * math.cos(theta)
        )

        x_upper.append(xu)
        y_upper.append(yu)
        x_lower.append(xl)
        y_lower.append(yl)

    return Airfoil(
        name=f"NACA {code}",
        x_upper=np.array(x_upper) * chord,
        y_upper=np.array(y_upper) * chord,
        x_lower=np.array(x_lower) * chord,
        y_lower=np.array(y_lower) * chord,
    )




def monotone_surface(x, y, tol=1e-9):
    """
    Ordena las coordenadas de una superficie (x, y) de modo que x sea
    estrictamente creciente, descartando puntos duplicados o con separacion
    en x menor o igual a la tolerancia especificada.

    Parametros
    ----------
    x : array-like
        Coordenadas x de la superficie.
    y : array-like
        Coordenadas y de la superficie.
    tol : float, optional
        Tolerancia minima de separacion entre puntos consecutivos en x.
        Por defecto 1e-9.

    Retorna
    -------
    x_mono : ndarray
        Coordenadas x ordenadas y estrictamente crecientes.
    y_mono : ndarray
        Coordenadas y correspondientes a los puntos filtrados.
    """
    if isinstance(x, (bool, np.bool_)) or isinstance(y, (bool, np.bool_)):
        raise TypeError("x and y must be numeric arrays, not booleans")

    if any(isinstance(v, (bool, np.bool_)) for v in (x if not isinstance(x, np.ndarray) else x.flat)):
        raise TypeError("x must contain numeric values, not booleans")
    if any(isinstance(v, (bool, np.bool_)) for v in (y if not isinstance(y, np.ndarray) else y.flat)):
        raise TypeError("y must contain numeric values, not booleans")

    try:
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("x and y must be convertible to numeric arrays")

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y must be 1D arrays")

    if len(x_arr) != len(y_arr):
        raise ValueError("x and y must have the same length")

    if len(x_arr) < 2:
        raise ValueError("At least two points are required")

    order = np.argsort(x_arr, kind="mergesort")
    x_sorted = x_arr[order]
    y_sorted = y_arr[order]

    keep = [0]
    for i in range(1, len(x_sorted)):
        if x_sorted[i] > x_sorted[keep[-1]] + tol:
            keep.append(i)

    return np.asarray(x_sorted[keep]), np.asarray(y_sorted[keep])
