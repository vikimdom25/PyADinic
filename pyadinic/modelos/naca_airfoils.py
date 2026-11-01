import math 
import numpy as np
from pyadinic.core.dataclass import Airfoil

def parse_naca4(chain: str):

    if not isinstance(chain, str):
        raise TypeError("string of numbers as argument needed")

    if "-" in chain:
        parts = chain.split("-")
        if len(parts) != 2:
            raise TypeError("Invalid NACA modified format")
        base_chain, mod_chain = parts
        if len(base_chain) != 4 or len(mod_chain) != 2:
            raise TypeError("Base code must be 4 digits and modification 2 digits")

        try:
            for digit in base_chain + mod_chain:
                float(digit)
        except ValueError:
            raise TypeError("String not convertible in a numeric value")

        first = base_chain[0]
        second = base_chain[1]
        last = base_chain[2:]

        m = float(first) / 100
        p = float(second) / 10
        t = float(last) / 100

        if not (0 <= m <= 0.09):
            raise ValueError("Max camber out of range")
        if m == 0 and p == 0:
            pass
        elif not (0.1 <= p <= 0.9):
            raise ValueError("Camber max position out of range")
        if not (0 <= t <= 0.99):
            raise ValueError("Tickness out of range")

        le_index = float(mod_chain[0])
        xmaxt = float(mod_chain[1]) / 10.0
        return m, p, t, le_index, xmaxt

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


def thickness_distribution_modified(x, t, le_index=6.0, xmaxt=0.3):
    """
    Calcula la distribucion de semiespesor en dos tramos para perfiles NACA modificados
    (serie de 4 y 5 digitos con sufijo -ab).

    Parametros
    ----------
    x : float o array-like
        Coordenada de cuerda adimensional en [0, 1].
    t : float
        Espesor maximo adimensional t/c (0 < t <= 1).
    le_index : float, optional
        Indice del radio de borde de ataque (primer digito tras el guion).
        Por defecto 6.0 (estandar). 0 indica borde de ataque afilado.
    xmaxt : float, optional
        Posicion del espesor maximo en fraccion de cuerda (segundo digito / 10).
        Por defecto 0.3.

    Retorna
    -------
    float o ndarray
        Semiespesor y_t(x).
    """
    if isinstance(x, (bool, np.bool_)) or isinstance(t, (bool, np.bool_)) or isinstance(le_index, (bool, np.bool_)) or isinstance(xmaxt, (bool, np.bool_)):
        raise TypeError("Inputs must be numeric, not boolean")

    if not isinstance(t, (int, float, np.integer, np.floating)):
        raise TypeError("Thickness t must be numeric")
    if not isinstance(le_index, (int, float, np.integer, np.floating)):
        raise TypeError("le_index must be numeric")
    if not isinstance(xmaxt, (int, float, np.integer, np.floating)):
        raise TypeError("xmaxt must be numeric")

    t = float(t)
    le_index = float(le_index)
    xmaxt = float(xmaxt)

    if not (0.0 < t <= 1.0):
        raise ValueError("Thickness t must be in (0, 1]")
    if le_index < 0.0:
        raise ValueError("le_index must be non-negative")
    if not (0.05 <= xmaxt <= 0.95):
        raise ValueError("xmaxt must be between 0.05 and 0.95")

    A = [3.48e-5, 2.3076628, -10.127712, 19.961478, -10.420597]
    d1 = sum(A[i] * (xmaxt**i) for i in range(5))

    rle = (1.1019 / 36.0) * ((t * le_index) ** 2)
    a0 = (0.2 / t) * math.sqrt(2.0 * rle) if t > 0.0 else 0.0

    omxmt = 1.0 - xmaxt
    omxmsq = omxmt * omxmt

    a3 = (0.1 / (xmaxt**3)) + ((d1 * omxmt - 0.294) / (xmaxt * omxmsq)) - ((3.0 / 8.0) * a0 / (xmaxt**2.5))
    a2 = (-0.1 / (xmaxt**2)) + (0.5 * a0 / (xmaxt**1.5)) - (2.0 * a3 * xmaxt)
    a1 = (-0.5 * a0 / math.sqrt(xmaxt)) - (2.0 * a2 * xmaxt) - (3.0 * a3 * (xmaxt**2))

    d3 = ((3.0 * d1) - (0.588 / omxmt)) / (3.0 * omxmsq)
    d2 = (-1.5 * omxmt * d3) - ((0.5 * d1) / omxmt)
    d0 = 0.002

    def _eval_single(xv):
        if isinstance(xv, (bool, np.bool_)):
            raise TypeError("x cannot contain boolean values")
        if not (0.0 <= xv <= 1.0):
            raise ValueError("Normalized Chord, x out of range")
        if xv == 0.0:
            return 0.0
        elif xv < xmaxt:
            srx = math.sqrt(xv)
            yt_base = a0 * srx + xv * (a1 + xv * (a2 + xv * a3))
        else:
            xx = 1.0 - xv
            yt_base = d0 + xx * (d1 + xx * (d2 + xx * d3))
        return 5.0 * t * yt_base

    if np.isscalar(x):
        return _eval_single(float(x))
    else:
        x_arr = np.asarray(x, dtype=float)
        return np.vectorize(_eval_single, otypes=[float])(x_arr)


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

    
def naca4(code, chord=1.0, n_points=100, spacing_type="cosine", thickness_func=None):

    parsed = parse_naca4(code)
    if len(parsed) == 5:
        m, p, t, le_index, xmaxt = parsed
        is_modified = True
    else:
        m, p, t = parsed
        is_modified = False
        le_index, xmaxt = 6.0, 0.3

    x = spacing(n_points, spacing_type=spacing_type)

    x_upper = []
    y_upper = []
    x_lower = []
    y_lower = []

    if chord <= 0:
        raise ValueError("Chord must be positive")

    for x_value in x:
        y_c = camber_line(x_value, m, p)
        if thickness_func is not None:
            try:
                y_t = thickness_func(x_value, t)
            except TypeError:
                y_t = thickness_func(x_value)
        elif is_modified:
            y_t = thickness_distribution_modified(x_value, t, le_index=le_index, xmaxt=xmaxt)
        else:
            y_t = thickness_distribution(x_value, t)

        slope = camber_line_slope(x_value, m, p)
        theta = math.atan(slope)

        xu = x_value - y_t * math.sin(theta)
        yu = y_c + y_t * math.cos(theta)
        xl = x_value + y_t * math.sin(theta)
        yl = y_c - y_t * math.cos(theta)

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

    if "-" in chain:
        parts = chain.split("-")
        if len(parts) != 2:
            raise TypeError("Invalid NACA modified format")
        base_chain, mod_chain = parts
        if len(base_chain) != 5 or len(mod_chain) != 2:
            raise TypeError("Base code must be 5 digits and modification 2 digits")
        try:
            for digit in mod_chain:
                float(digit)
        except ValueError:
            raise TypeError("String not convertible in a numeric value")
        le_index = float(mod_chain[0])
        xmaxt = float(mod_chain[1]) / 10.0
        parsed_base = parse_naca5(base_chain)
        return parsed_base + (le_index, xmaxt)

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

def naca5(code, chord=1.0, n_points=100, spacing_type="cosine", thickness_func=None):

    parsed = parse_naca5(code)
    if len(parsed) == 9:
        (
            design_lift_coefficient,
            max_camber_position,
            camber_type,
            t,
            r,
            k1,
            k2_ratio,
            le_index,
            xmaxt,
        ) = parsed
        is_modified = True
    else:
        (
            design_lift_coefficient,
            max_camber_position,
            camber_type,
            t,
            r,
            k1,
            k2_ratio,
        ) = parsed
        is_modified = False
        le_index, xmaxt = 6.0, 0.3

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

        if thickness_func is not None:
            try:
                y_t = thickness_func(x_value, t)
            except TypeError:
                y_t = thickness_func(x_value)
        elif is_modified:
            y_t = thickness_distribution_modified(x_value, t, le_index=le_index, xmaxt=xmaxt)
        else:
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



def airfoil_from_coordinates(name: str, x, y, x_lower=None, y_lower=None):
    """
    Crea un objeto Airfoil a partir de coordenadas arbitrarias.

    Soporta dos modalidades de entrada:
    1. Superficies separadas: se pasan x (extrados) e y (extrados), junto con
       x_lower (intrados) e y_lower (intrados).
    2. Contorno cerrado continuo: se pasan x e y conteniendo todos los puntos
       (ej. formato Selig desde el borde de salida superior, pasando por el
       borde de ataque, hasta el borde de salida inferior). La funcion divide
       automaticamente el contorno en extrados e intrados en el borde de ataque
       (x minimo) y garantiza que ambas superficies vayan de borde de ataque a salida.

    Parametros
    ----------
    name : str
        Nombre o identificador del perfil.
    x : array-like
        Coordenadas x del perfil o de la superficie superior.
    y : array-like
        Coordenadas y del perfil o de la superficie superior.
    x_lower : array-like, optional
        Coordenadas x de la superficie inferior si se suministra por separado.
    y_lower : array-like, optional
        Coordenadas y de la superficie inferior si se suministra por separado.

    Retorna
    -------
    Airfoil
        Instancia de Airfoil con x_upper, y_upper, x_lower, y_lower ordenados
        de borde de ataque a borde de salida.
    """
    if not isinstance(name, str):
        raise TypeError("Airfoil name must be a string")

    if (x_lower is None) != (y_lower is None):
        raise ValueError("Both x_lower and y_lower must be provided together, or both omitted")

    def _check_no_bool(arr, label):
        if isinstance(arr, (bool, np.bool_)):
            raise TypeError(f"{label} cannot be a boolean")
        if any(isinstance(v, (bool, np.bool_)) for v in (arr if not isinstance(arr, np.ndarray) else arr.flat)):
            raise TypeError(f"{label} contains boolean values")

    _check_no_bool(x, "x")
    _check_no_bool(y, "y")

    if x_lower is not None:
        _check_no_bool(x_lower, "x_lower")
        _check_no_bool(y_lower, "y_lower")

        xu = np.asarray(x, dtype=float)
        yu = np.asarray(y, dtype=float)
        xl = np.asarray(x_lower, dtype=float)
        yl = np.asarray(y_lower, dtype=float)

        if xu.ndim != 1 or yu.ndim != 1 or xl.ndim != 1 or yl.ndim != 1:
            raise ValueError("All coordinate arrays must be 1D")
        if len(xu) != len(yu) or len(xl) != len(yl):
            raise ValueError("Coordinate array lengths must match for each surface")
        if len(xu) < 2 or len(xl) < 2:
            raise ValueError("Each surface must contain at least 2 points")

        # Asegurar orden LE -> TE (x creciente)
        if xu[0] > xu[-1]:
            xu = xu[::-1]
            yu = yu[::-1]
        if xl[0] > xl[-1]:
            xl = xl[::-1]
            yl = yl[::-1]

        return Airfoil(name=name, x_upper=xu, y_upper=yu, x_lower=xl, y_lower=yl)

    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    if x_arr.ndim != 1 or y_arr.ndim != 1:
        raise ValueError("x and y arrays must be 1D")
    if len(x_arr) != len(y_arr):
        raise ValueError("x and y arrays must have the same length")
    if len(x_arr) < 4:
        raise ValueError("At least 4 points are required for a closed airfoil contour")

    idx_le = int(np.argmin(x_arr))

    if idx_le == 0 or idx_le == len(x_arr) - 1:
        idx_te = int(np.argmax(x_arr))
        branch1_x = x_arr[:idx_te + 1]
        branch1_y = y_arr[:idx_te + 1]
        branch2_x = x_arr[idx_te:]
        branch2_y = y_arr[idx_te:]
    else:
        branch1_x = x_arr[:idx_le + 1]
        branch1_y = y_arr[:idx_le + 1]
        branch2_x = x_arr[idx_le:]
        branch2_y = y_arr[idx_le:]

    if branch1_x[0] > branch1_x[-1]:
        branch1_x = branch1_x[::-1]
        branch1_y = branch1_y[::-1]
    if branch2_x[0] > branch2_x[-1]:
        branch2_x = branch2_x[::-1]
        branch2_y = branch2_y[::-1]

    if np.mean(branch1_y) >= np.mean(branch2_y):
        xu, yu = branch1_x, branch1_y
        xl, yl = branch2_x, branch2_y
    else:
        xu, yu = branch2_x, branch2_y
        xl, yl = branch1_x, branch1_y

    return Airfoil(name=name, x_upper=xu, y_upper=yu, x_lower=xl, y_lower=yl)


def airfoil_from_selig(source, normalize=True):
    """
    Carga y normaliza un perfil aerodinamico en formato Selig .dat desde un
    archivo en disco, ruta, objeto de archivo o cadena multilineal.

    El formato Selig presenta el nombre en la primera linea no vacia, seguido
    de pares de coordenadas x y desde el borde de salida del extrados (x ≈ 1),
    pasando por el borde de ataque (x ≈ 0), hasta el borde de salida del intrados (x ≈ 1).

    Parametros
    ----------
    source : str, os.PathLike, or file-like
        Ruta del archivo, cadena de texto o flujo de entrada.
    normalize : bool, optional
        Si True, escala la cuerda a [0, 1] y centra el borde de ataque en 0.
        Por defecto True.

    Retorna
    -------
    Airfoil
        Instancia de Airfoil creada y normalizada.
    """
    import os

    if hasattr(source, "read"):
        content_src = source.read()
        if isinstance(content_src, bytes):
            content_src = content_src.decode("utf-8", errors="ignore")
    elif isinstance(source, (str, os.PathLike)):
        str_src = str(source)
        if (chr(10) in str_src) or (chr(13) in str_src):
            content_src = str_src
        else:
            with open(str_src, "r", encoding="utf-8", errors="ignore") as f:
                content_src = f.read()
    else:
        raise TypeError("source must be a file path, file-like object, or multiline string")

    raw_lines = [line.strip() for line in content_src.splitlines()]
    non_empty = [line for line in raw_lines if line and not line.startswith(("#", "//"))]

    if not non_empty:
        raise ValueError("No data found in Selig airfoil source")

    name = non_empty[0]
    coord_lines = non_empty[1:]

    x_pts = []
    y_pts = []
    for line in coord_lines:
        tokens = line.split()
        if len(tokens) >= 2:
            try:
                x_val = float(tokens[0])
                y_val = float(tokens[1])
                x_pts.append(x_val)
                y_pts.append(y_val)
            except ValueError:
                continue

    if len(x_pts) < 4:
        raise ValueError(f"Insufficient coordinate points found in Selig source: {len(x_pts)}")

    x_arr = np.asarray(x_pts, dtype=float)
    y_arr = np.asarray(y_pts, dtype=float)

    if normalize:
        x_min = np.min(x_arr)
        x_max = np.max(x_arr)
        chord = x_max - x_min
        if chord > 0.0:
            x_arr = (x_arr - x_min) / chord
            y_arr = y_arr / chord

    return airfoil_from_coordinates(name, x_arr, y_arr)
