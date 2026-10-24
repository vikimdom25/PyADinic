import math
import numpy as np

from pyadinic.core.dataclass import ThinAirfoilState
from pyadinic.numerical_methods.integration import integrate_gauss_legendre
from .naca_airfoils import (
    parse_naca4,
    parse_naca5,
    camber_line_slope,
    camber_line_slope_5,
)


def _resolve_slope_function(slope_func=None, naca=None):
    """
    Resuelve la función de pendiente dy_c/dx a partir de una función dada
    o de la designación NACA de 4 o 5 dígitos.
    """
    if slope_func is not None and naca is not None:
        raise ValueError("Provide either slope_func or naca, not both")

    if slope_func is None and naca is None:
        raise ValueError("Must provide either slope_func or naca")

    if naca is not None:
        if not isinstance(naca, str):
            raise TypeError("naca must be a string code")

        code = naca.strip().upper().replace("NACA", "").strip()

        if len(code) == 4:
            m, p, _ = parse_naca4(code)
            return lambda x: camber_line_slope(x, m, p)
        elif len(code) == 5:
            _, _, third, _, r, k1, k2_ratio = parse_naca5(code)
            reflex = third == 1
            return lambda x: camber_line_slope_5(x, r, k1, reflex=reflex, k2_ratio=k2_ratio)
        else:
            raise ValueError("naca code must be 4 or 5 digits")

    if not callable(slope_func):
        raise TypeError("slope_func must be callable")

    return slope_func


def fourier_coefficients(slope_func=None, naca=None, n_coefficients=3, n_points=50):
    """
    Calcula los coeficientes de Fourier de la teoría de perfiles delgados
    mediante cuadratura de Gauss-Legendre:
      A0_camber = (1 / pi) * integral_0^pi (dy_c/dx) dtheta
      A_n = (2 / pi) * integral_0^pi (dy_c/dx) * cos(n * theta) dtheta
    
    Retorna:
      (a0_camber, [A1, A2, ..., An])
    """
    func = _resolve_slope_function(slope_func=slope_func, naca=naca)

    if not isinstance(n_coefficients, int) or isinstance(n_coefficients, bool):
        raise TypeError("n_coefficients must be an integer")
    if n_coefficients < 1:
        raise ValueError("n_coefficients must be at least 1")

    if not isinstance(n_points, int) or isinstance(n_points, bool):
        raise TypeError("n_points must be an integer")
    if n_points < 4:
        raise ValueError("n_points must be at least 4")

    def slope_theta(theta):
        x = 0.5 * (1.0 - math.cos(theta))
        return func(x)

    # A0_camber
    a0_camber = (1.0 / math.pi) * integrate_gauss_legendre(
        slope_theta, 0.0, math.pi, n_points
    )

    # An para n >= 1
    an_list = []
    for n in range(1, n_coefficients + 1):
        def integrand(theta, n_val=n):
            return slope_theta(theta) * math.cos(n_val * theta)

        val = (2.0 / math.pi) * integrate_gauss_legendre(
            integrand, 0.0, math.pi, n_points
        )
        an_list.append(val)

    return a0_camber, an_list


def thin_airfoil_alpha_0(slope_func=None, naca=None, deg=False, n_points=50):
    """
    Calcula el ángulo de ataque de sustentación nula (alpha_0):
      alpha_0 = A0_camber - 0.5 * A1
              = - (1 / pi) * integral_0^pi (dy_c/dx) * (cos(theta) - 1) dtheta
    """
    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    a0_camber, an_list = fourier_coefficients(
        slope_func=slope_func, naca=naca, n_coefficients=1, n_points=n_points
    )
    a1 = an_list[0]

    alpha_0_rad = a0_camber - 0.5 * a1

    return math.degrees(alpha_0_rad) if deg else alpha_0_rad


def thin_airfoil_cl(alpha, slope_func=None, naca=None, deg=False, n_points=50):
    """
    Calcula el coeficiente de sustentación 2D según la teoría de perfiles delgados:
      cl = 2 * pi * (alpha - alpha_0)
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    alpha_rad = math.radians(alpha) if deg else float(alpha)
    alpha_0_rad = thin_airfoil_alpha_0(slope_func=slope_func, naca=naca, deg=False, n_points=n_points)

    return 2.0 * math.pi * (alpha_rad - alpha_0_rad)


def thin_airfoil_cm_c4(slope_func=None, naca=None, n_points=50):
    """
    Calcula el coeficiente de momento de cabeceo alrededor de c/4:
      cm_c/4 = (pi / 4) * (A2 - A1)
    Es independiente del ángulo de ataque alpha.
    """
    _, an_list = fourier_coefficients(
        slope_func=slope_func, naca=naca, n_coefficients=2, n_points=n_points
    )
    a1 = an_list[0]
    a2 = an_list[1]

    return (math.pi / 4.0) * (a2 - a1)


def thin_airfoil_cm_le(alpha, slope_func=None, naca=None, deg=False, n_points=50):
    """
    Calcula el coeficiente de momento de cabeceo alrededor del borde de ataque (LE):
      cm_le = cm_c/4 - 0.25 * cl
    """
    cl = thin_airfoil_cl(alpha, slope_func=slope_func, naca=naca, deg=deg, n_points=n_points)
    cm_c4 = thin_airfoil_cm_c4(slope_func=slope_func, naca=naca, n_points=n_points)

    return cm_c4 - 0.25 * cl


def thin_airfoil_center_of_pressure(alpha, slope_func=None, naca=None, deg=False, n_points=50):
    """
    Calcula la posición adimensional del centro de presiones (x_cp / c):
      x_cp / c = 0.25 - (cm_c/4 / cl) = 0.25 * [1 - (4 * cm_c/4 / cl)]
    """
    cl = thin_airfoil_cl(alpha, slope_func=slope_func, naca=naca, deg=deg, n_points=n_points)
    cm_c4 = thin_airfoil_cm_c4(slope_func=slope_func, naca=naca, n_points=n_points)

    if abs(cl) < 1e-12:
        return 0.25

    return 0.25 - (cm_c4 / cl)


def thin_airfoil_delta_cp(x, alpha, slope_func=None, naca=None, deg=False, n_coefficients=10, n_points=50):
    """
    Calcula la diferencia de coeficiente de presión (salto de presión) entre intradós y extradós:
      Delta Cp(theta) = 4 * [ A0 * (1 + cos(theta)) / sin(theta) + sum_{n=1}^N An * sin(n * theta) ]
    
    Parámetro x: posición adimensional en la cuerda (0 < x < 1), puede ser escalar o ndarray.
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    alpha_rad = math.radians(alpha) if deg else float(alpha)

    a0_camber, an_list = fourier_coefficients(
        slope_func=slope_func, naca=naca, n_coefficients=n_coefficients, n_points=n_points
    )
    a0 = alpha_rad - a0_camber

    def compute_single_cp(x_val):
        if not (0 <= x_val <= 1):
            raise ValueError("x must be in [0, 1]")
        if x_val == 0.0:
            return float("inf") if a0 != 0 else 0.0
        if x_val == 1.0:
            return 0.0

        theta = math.acos(1.0 - 2.0 * x_val)
        sin_th = math.sin(theta)
        cos_th = math.cos(theta)

        val = a0 * (1.0 + cos_th) / sin_th
        for n, an in enumerate(an_list, start=1):
            val += an * math.sin(n * theta)

        return 4.0 * val

    if isinstance(x, (np.ndarray, list, tuple)):
        x_arr = np.asarray(x, dtype=float)
        return np.array([compute_single_cp(xi) for xi in x_arr])

    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError("x must be numeric")

    return compute_single_cp(x)


def thin_airfoil_theory(alpha, slope_func=None, naca=None, deg=False, n_points=50):
    """
    Resuelve la teoría completa de perfiles delgados para un ángulo de ataque dado y
    retorna un objeto ThinAirfoilState con todos los parámetros aerodinámicos calculados:
      - alpha: ángulo de ataque en radianes
      - cl: coeficiente de sustentación
      - cm_c4: coeficiente de momento en cuarto de cuerda
      - cm_le: coeficiente de momento en borde de ataque
      - alpha_0: ángulo de sustentación nula en radianes
      - lift_slope: pendiente de sustentación dcl/dalpha = 2 * pi
      - center_of_pressure: posición adimensional x_cp / c
      - a0, a1, a2: primeros coeficientes de Fourier
    """
    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    alpha_rad = math.radians(alpha) if deg else float(alpha)

    a0_camber, an_list = fourier_coefficients(
        slope_func=slope_func, naca=naca, n_coefficients=2, n_points=n_points
    )
    a1 = an_list[0]
    a2 = an_list[1]
    a0 = alpha_rad - a0_camber

    alpha_0_rad = a0_camber - 0.5 * a1
    cl = 2.0 * math.pi * (alpha_rad - alpha_0_rad)
    cm_c4 = (math.pi / 4.0) * (a2 - a1)
    cm_le = cm_c4 - 0.25 * cl

    if abs(cl) < 1e-12:
        x_cp = 0.25
    else:
        x_cp = 0.25 - (cm_c4 / cl)

    return ThinAirfoilState(
        alpha=alpha_rad,
        cl=cl,
        cm_c4=cm_c4,
        cm_le=cm_le,
        alpha_0=alpha_0_rad,
        lift_slope=2.0 * math.pi,
        center_of_pressure=x_cp,
        a0=a0,
        a1=a1,
        a2=a2,
    )


def thin_airfoil_naca4(code, alpha=0.0, deg=False, n_points=50):
    """
    Atajo directo para calcular la teoría de perfiles delgados en perfiles NACA 4 dígitos.
    """
    return thin_airfoil_theory(alpha=alpha, naca=code, deg=deg, n_points=n_points)


def thin_airfoil_naca5(code, alpha=0.0, deg=False, n_points=50):
    """
    Atajo directo para calcular la teoría de perfiles delgados en perfiles NACA 5 dígitos.
    """
    return thin_airfoil_theory(alpha=alpha, naca=code, deg=deg, n_points=n_points)
