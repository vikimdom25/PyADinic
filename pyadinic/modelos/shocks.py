import math
from pyadinic.core.dataclass import NormalShockState, ObliqueShockState
from pyadinic.numerical_methods.solvers import solve_brent


def _validate_numeric(*args):
    for arg in args:
        if not isinstance(arg, (int, float)) or isinstance(arg, bool):
            raise TypeError("All numeric arguments must be int or float (not bool)")


# =============================================================================
# ONDAS DE CHOQUE NORMAL (NORMAL SHOCKS)
# =============================================================================

def normal_shock_mach2(mach1: float, gamma: float = 1.4) -> float:
    """Calcula el número de Mach aguas abajo (M2) de una onda de choque normal."""
    _validate_numeric(mach1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Upstream Mach number M1 must be >= 1.0 for a normal shock")

    num = 1.0 + ((gamma - 1.0) / 2.0) * (mach1**2)
    den = gamma * (mach1**2) - ((gamma - 1.0) / 2.0)
    return math.sqrt(num / den)


def normal_shock_pressure_ratio(mach1: float, gamma: float = 1.4) -> float:
    """Calcula la relación de presión estática (P2/P1) a través de un choque normal."""
    _validate_numeric(mach1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Upstream Mach number M1 must be >= 1.0 for a normal shock")

    return 1.0 + ((2.0 * gamma) / (gamma + 1.0)) * ((mach1**2) - 1.0)


def normal_shock_density_ratio(mach1: float, gamma: float = 1.4) -> float:
    """Calcula la relación de densidad estática (rho2/rho1) a través de un choque normal."""
    _validate_numeric(mach1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Upstream Mach number M1 must be >= 1.0 for a normal shock")

    num = (gamma + 1.0) * (mach1**2)
    den = 2.0 + (gamma - 1.0) * (mach1**2)
    return num / den


def normal_shock_temperature_ratio(mach1: float, gamma: float = 1.4) -> float:
    """Calcula la relación de temperatura estática (T2/T1) a través de un choque normal."""
    p_ratio = normal_shock_pressure_ratio(mach1, gamma)
    rho_ratio = normal_shock_density_ratio(mach1, gamma)
    return p_ratio / rho_ratio


def normal_shock_total_pressure_ratio(mach1: float, gamma: float = 1.4) -> float:
    """Calcula la relación de presión total (P02/P01) a través de un choque normal (pérdida de presión de estancamiento)."""
    _validate_numeric(mach1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Upstream Mach number M1 must be >= 1.0 for a normal shock")

    term1 = (((gamma + 1.0) / 2.0) * (mach1**2)) / (1.0 + ((gamma - 1.0) / 2.0) * (mach1**2))
    term2 = ((2.0 * gamma * (mach1**2)) - (gamma - 1.0)) / (gamma + 1.0)

    return (term1 ** (gamma / (gamma - 1.0))) * (term2 ** (-1.0 / (gamma - 1.0)))


def rayleigh_pitot_ratio(mach1: float, gamma: float = 1.4) -> float:
    """Calcula la relación de Rayleigh-Pitot (P02/P1) medida por un tubo Pitot en flujo supersónico."""
    _validate_numeric(mach1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Upstream Mach number M1 must be >= 1.0 for supersonic Pitot formula")

    term1 = ((gamma + 1.0) / 2.0) * (mach1**2)
    term2 = ((2.0 * gamma * (mach1**2)) - (gamma - 1.0)) / (gamma + 1.0)

    return (term1 ** (gamma / (gamma - 1.0))) * (term2 ** (-1.0 / (gamma - 1.0)))


def normal_shock_relations(mach1: float, gamma: float = 1.4) -> NormalShockState:
    """Calcula el estado completo de relaciones de salto a través de una onda de choque normal."""
    m2 = normal_shock_mach2(mach1, gamma)
    p2_p1 = normal_shock_pressure_ratio(mach1, gamma)
    rho2_rho1 = normal_shock_density_ratio(mach1, gamma)
    t2_t1 = p2_p1 / rho2_rho1
    p02_p01 = normal_shock_total_pressure_ratio(mach1, gamma)
    p02_p1 = rayleigh_pitot_ratio(mach1, gamma)

    return NormalShockState(
        mach_1=float(mach1),
        mach_2=m2,
        pressure_ratio=p2_p1,
        temperature_ratio=t2_t1,
        density_ratio=rho2_rho1,
        total_pressure_ratio=p02_p01,
        pitot_pressure_ratio=p02_p1,
    )


def mach1_from_normal_shock_pressure_ratio(p2_p1: float, gamma: float = 1.4) -> float:
    """Calcula el Mach 1 inicial a partir de la relación de presiones P2/P1."""
    _validate_numeric(p2_p1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if p2_p1 < 1.0:
        raise ValueError("Pressure ratio across normal shock must be >= 1.0")

    return math.sqrt(((gamma + 1.0) / (2.0 * gamma)) * (p2_p1 - 1.0) + 1.0)


def mach1_from_rayleigh_pitot(p02_p1: float, gamma: float = 1.4) -> float:
    """Invierte la fórmula de Rayleigh-Pitot para obtener M1 a partir de la relación P02/P1."""
    _validate_numeric(p02_p1, gamma)
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    
    # Para M1 = 1, P02/P1 = ((gamma+1)/2)^(gamma/(gamma-1))
    min_ratio = ((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))
    if p02_p1 < min_ratio:
        raise ValueError(
            f"Pitot pressure ratio P02/P1 must be >= {min_ratio:.4f} for supersonic flow"
        )

    def residual(m):
        return rayleigh_pitot_ratio(m, gamma) - p02_p1

    return solve_brent(residual, 1.0, 50.0, tolerance=1e-10)


# =============================================================================
# ONDAS DE CHOQUE OBLICUAS (OBLIQUE SHOCKS)
# =============================================================================

def theta_from_beta_mach(beta: float, mach1: float, gamma: float = 1.4, use_degrees: bool = False) -> float:
    """Calcula el ángulo de deflexión theta a partir del ángulo de choque beta y Mach 1 (ecuación theta-beta-Mach)."""
    _validate_numeric(beta, mach1, gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 < 1.0:
        raise ValueError("Mach 1 must be >= 1.0 for oblique shock")

    beta_rad = math.radians(beta) if use_degrees else float(beta)
    mu_rad = math.asin(1.0 / mach1)

    if beta_rad < mu_rad or beta_rad > (math.pi / 2.0):
        raise ValueError(f"Shock angle beta must be between Mach angle ({math.degrees(mu_rad):.2f}°) and 90°")

    num = (mach1**2) * (math.sin(beta_rad)**2) - 1.0
    den = (mach1**2) * (gamma + math.cos(2.0 * beta_rad)) + 2.0
    tan_theta = 2.0 * (1.0 / math.tan(beta_rad)) * (num / den)
    theta_rad = math.atan(tan_theta)

    return math.degrees(theta_rad) if use_degrees else theta_rad


def max_deflection_angle(mach1: float, gamma: float = 1.4, use_degrees: bool = False) -> tuple[float, float]:
    """Calcula analíticamente la deflexión máxima theta_max y el ángulo de choque beta asociado para Mach 1."""
    _validate_numeric(mach1, gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 <= 1.0:
        raise ValueError("Mach 1 must be > 1.0 for oblique shock")

    m2 = mach1**2
    term1 = ((gamma + 1.0) / 4.0) * m2 - 1.0
    term2 = math.sqrt((gamma + 1.0) * (1.0 + ((gamma - 1.0) / 2.0) * m2 + ((gamma + 1.0) / 16.0) * (m2**2)))
    sin2_beta = (term1 + term2) / (gamma * m2)
    sin2_beta = max(0.0, min(1.0, sin2_beta))
    beta_max_rad = math.asin(math.sqrt(sin2_beta))

    theta_max_rad = theta_from_beta_mach(beta_max_rad, mach1, gamma, use_degrees=False)

    if use_degrees:
        return math.degrees(theta_max_rad), math.degrees(beta_max_rad)
    return theta_max_rad, beta_max_rad


def oblique_shock_beta(
    mach1: float,
    theta: float,
    gamma: float = 1.4,
    branch: str = "weak",
    use_degrees: bool = False,
) -> float:
    """Calcula el ángulo de choque beta para una deflexión theta dada.

    Args:
        mach1: Mach aguas arriba (M1 > 1).
        theta: Ángulo de deflexión del flujo.
        gamma: Relación de calores específicos (default 1.4).
        branch: 'weak' (choque débil, rama usual) o 'strong' (choque fuerte).
        use_degrees: Si True, interpreta theta y retorna beta en grados sexagesimales.
    """
    _validate_numeric(mach1, theta, gamma)
    if not isinstance(use_degrees, bool):
        raise TypeError("use_degrees must be a boolean")
    if gamma <= 1:
        raise ValueError("Gamma must be greater than 1")
    if mach1 <= 1.0:
        raise ValueError("Mach 1 must be > 1.0 for oblique shock")
    if branch not in ("weak", "strong"):
        raise ValueError("branch must be 'weak' or 'strong'")

    theta_rad = math.radians(theta) if use_degrees else float(theta)
    if theta_rad < 0:
        raise ValueError("Deflection angle theta must be >= 0")

    theta_max_rad, beta_max_rad = max_deflection_angle(mach1, gamma, use_degrees=False)

    if theta_rad > theta_max_rad:
        raise ValueError(
            f"Deflection theta exceeds maximum allowed theta_max = {math.degrees(theta_max_rad):.2f}° for M = {mach1}. "
            "Shock is detached."
        )

    if theta_rad == 0.0:
        mu_rad = math.asin(1.0 / mach1)
        if branch == "weak":
            return math.degrees(mu_rad) if use_degrees else mu_rad
        else:
            return 90.0 if use_degrees else (math.pi / 2.0)

    if abs(theta_rad - theta_max_rad) < 1e-9:
        return math.degrees(beta_max_rad) if use_degrees else beta_max_rad

    def residual(b):
        return theta_from_beta_mach(b, mach1, gamma, use_degrees=False) - theta_rad

    mu_rad = math.asin(1.0 / mach1)

    if branch == "weak":
        beta_rad = solve_brent(residual, mu_rad + 1e-9, beta_max_rad, tolerance=1e-10)
    else:
        beta_rad = solve_brent(residual, beta_max_rad, (math.pi / 2.0) - 1e-9, tolerance=1e-10)

    return math.degrees(beta_rad) if use_degrees else beta_rad


def oblique_shock_relations(
    mach1: float,
    beta: float = None,
    theta: float = None,
    gamma: float = 1.4,
    branch: str = "weak",
    use_degrees: bool = False,
) -> ObliqueShockState:
    """Calcula el estado termodinámico completo a través de una onda de choque oblicua.

    Se debe especificar 'beta' o 'theta' (uno de los dos).
    """
    _validate_numeric(mach1, gamma)
    if beta is None and theta is None:
        raise ValueError("Must specify either 'beta' (shock angle) or 'theta' (deflection angle)")
    if beta is not None and theta is not None:
        raise ValueError("Specify either 'beta' or 'theta', not both")

    if beta is not None:
        _validate_numeric(beta)
        beta_rad = math.radians(beta) if use_degrees else float(beta)
        theta_rad = theta_from_beta_mach(beta_rad, mach1, gamma, use_degrees=False)
    else:
        _validate_numeric(theta)
        theta_rad = math.radians(theta) if use_degrees else float(theta)
        beta_rad = oblique_shock_beta(mach1, theta_rad, gamma, branch=branch, use_degrees=False)

    m1n = mach1 * math.sin(beta_rad)
    if m1n < 1.0:
        raise ValueError("Normal Mach component M1n must be >= 1.0 for a shock wave to exist")

    normal_state = normal_shock_relations(m1n, gamma)
    m2n = normal_state.mach_2
    m2 = m2n / math.sin(beta_rad - theta_rad)

    return ObliqueShockState(
        mach_1=float(mach1),
        mach_2=m2,
        wave_angle=math.degrees(beta_rad) if use_degrees else beta_rad,
        deflection_angle=math.degrees(theta_rad) if use_degrees else theta_rad,
        normal_mach_1=m1n,
        normal_mach_2=m2n,
        pressure_ratio=normal_state.pressure_ratio,
        temperature_ratio=normal_state.temperature_ratio,
        density_ratio=normal_state.density_ratio,
        total_pressure_ratio=normal_state.total_pressure_ratio,
    )
