import math
from pyadinic.core.dataclass import PrandtlMeyerState
from pyadinic.numerical_methods.solvers import solve_brent


def _validate_numeric(*args):
    for arg in args:
        if not isinstance(arg, (int, float)) or isinstance(arg, bool):
            raise TypeError("All numeric arguments must be int or float (not bool)")


def mach_angle(mach: float, use_degrees: bool = False) -> float:
    """Calcula el ángulo de Mach mu = arcsin(1/M)."""
    _validate_numeric(mach)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if mach < 1.0:
        raise ValueError("Mach number must be >= 1.0 to compute Mach angle")

    mu_rad = math.asin(1.0 / float(mach))
    return math.degrees(mu_rad) if use_degrees else mu_rad


def max_prandtl_meyer_angle(gamma: float = 1.4, use_degrees: bool = False) -> float:
    """Calcula el ángulo máximo teórico de expansión de Prandtl-Meyer cuando Mach -> infinito."""
    _validate_numeric(gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")

    nu_max_rad = (math.pi / 2.0) * (math.sqrt((gamma + 1.0) / (gamma - 1.0)) - 1.0)
    return math.degrees(nu_max_rad) if use_degrees else nu_max_rad


def prandtl_meyer_function(mach: float, gamma: float = 1.4, use_degrees: bool = False) -> float:
    """Calcula el ángulo de la función de Prandtl-Meyer nu(M)."""
    _validate_numeric(mach, gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach < 1.0:
        raise ValueError("Mach number must be >= 1.0 for Prandtl-Meyer function")

    if mach == 1.0:
        return 0.0

    m2_minus_1 = (mach**2) - 1.0
    factor = math.sqrt((gamma + 1.0) / (gamma - 1.0))
    term1 = factor * math.atan(math.sqrt(((gamma - 1.0) / (gamma + 1.0)) * m2_minus_1))
    term2 = math.atan(math.sqrt(m2_minus_1))
    nu_rad = term1 - term2

    return math.degrees(nu_rad) if use_degrees else nu_rad


def mach_from_prandtl_meyer(nu: float, gamma: float = 1.4, use_degrees: bool = False) -> float:
    """Calcula el número de Mach M correspondiente a un valor dado de la función de Prandtl-Meyer nu."""
    _validate_numeric(nu, gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")

    nu_rad = math.radians(nu) if use_degrees else float(nu)
    if nu_rad < 0:
        raise ValueError("Prandtl-Meyer angle nu must be >= 0")
    if nu_rad == 0.0:
        return 1.0

    nu_max_rad = max_prandtl_meyer_angle(gamma, use_degrees=False)
    if nu_rad >= nu_max_rad:
        raise ValueError(
            f"Prandtl-Meyer angle nu exceeds maximum theoretical turning angle ({math.degrees(nu_max_rad):.2f}°)"
        )

    def residual(m):
        return prandtl_meyer_function(m, gamma, use_degrees=False) - nu_rad

    return solve_brent(residual, 1.0, 100.0, tolerance=1e-10)


def prandtl_meyer_expansion(
    mach1: float,
    theta: float,
    gamma: float = 1.4,
    use_degrees: bool = False,
) -> PrandtlMeyerState:
    """Calcula las propiedades termodinámicas y cinemáticas tras una expansión isentrópica de Prandtl-Meyer.

    Args:
        mach1: Mach inicial aguas arriba (M1 >= 1).
        theta: Ángulo de deflexión/giro de expansión.
        gamma: Relación de calores específicos.
        use_degrees: Si True, interpreta theta y retorna ángulos en grados sexagesimales.

    Returns:
        PrandtlMeyerState con M1, M2, ángulos y relaciones de presión, temperatura y densidad.
    """
    _validate_numeric(mach1, theta, gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Upstream Mach M1 must be >= 1.0 for Prandtl-Meyer expansion")

    theta_rad = math.radians(theta) if use_degrees else float(theta)
    if theta_rad < 0:
        raise ValueError("Expansion turn angle theta must be >= 0")

    nu1_rad = prandtl_meyer_function(mach1, gamma, use_degrees=False)
    nu2_rad = nu1_rad + theta_rad

    nu_max_rad = max_prandtl_meyer_angle(gamma, use_degrees=False)
    if nu2_rad >= nu_max_rad:
        raise ValueError(
            f"Expanded turn angle leads to nu2 = {math.degrees(nu2_rad):.2f}°, which exceeds nu_max = {math.degrees(nu_max_rad):.2f}°."
        )

    mach2 = mach_from_prandtl_meyer(nu2_rad, gamma, use_degrees=False)

    mu1_rad = mach_angle(mach1, use_degrees=False)
    mu2_rad = mach_angle(mach2, use_degrees=False)

    # Relaciones isentrópicas P02 = P01, T02 = T01
    t0_t1 = 1.0 + ((gamma - 1.0) / 2.0) * (mach1**2)
    t0_t2 = 1.0 + ((gamma - 1.0) / 2.0) * (mach2**2)

    t2_t1 = t0_t1 / t0_t2
    p2_p1 = t2_t1 ** (gamma / (gamma - 1.0))
    rho2_rho1 = t2_t1 ** (1.0 / (gamma - 1.0))

    return PrandtlMeyerState(
        mach_1=float(mach1),
        mach_2=mach2,
        turn_angle=math.degrees(theta_rad) if use_degrees else theta_rad,
        nu_1=math.degrees(nu1_rad) if use_degrees else nu1_rad,
        nu_2=math.degrees(nu2_rad) if use_degrees else nu2_rad,
        mu_1=math.degrees(mu1_rad) if use_degrees else mu1_rad,
        mu_2=math.degrees(mu2_rad) if use_degrees else mu2_rad,
        pressure_ratio=p2_p1,
        temperature_ratio=t2_t1,
        density_ratio=rho2_rho1,
    )
