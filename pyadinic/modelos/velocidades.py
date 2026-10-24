import math
from pyadinic.core.constants import P0 as P0_DEFAULT, gamma_air as GAMMA_DEFAULT
from pyadinic.numerical_methods.solvers import solve_brent

RHO0_DEFAULT = 1.225


def _validate_numeric(*args):
    for arg in args:
        if not isinstance(arg, (int, float)) or isinstance(arg, bool):
            raise TypeError("Error, no numeric args")


# =============================================================================
# CONVERSIONES BÁSICAS (IAS / CAS / EAS / TAS / MACH)
# =============================================================================

def ias_to_cas(ias: float, position_error: float = 0.0, instrument_error: float = 0.0) -> float:
    """Convierte Indicated Airspeed (IAS) a Calibrated Airspeed (CAS) aplicando correcciones de error."""
    _validate_numeric(ias, position_error, instrument_error)
    if ias < 0:
        raise ValueError("IAS must be greater than or equal to zero")
    cas = ias + position_error + instrument_error
    if cas < 0:
        raise ValueError("Calibrated airspeed cannot be negative after error correction")
    return cas


def cas_to_ias(cas: float, position_error: float = 0.0, instrument_error: float = 0.0) -> float:
    """Convierte Calibrated Airspeed (CAS) a Indicated Airspeed (IAS)."""
    _validate_numeric(cas, position_error, instrument_error)
    if cas < 0:
        raise ValueError("CAS must be greater than or equal to zero")
    ias = cas - position_error - instrument_error
    if ias < 0:
        raise ValueError("Indicated airspeed cannot be negative after error correction")
    return ias


def tas_to_eas(tas: float, density: float, rho0: float = RHO0_DEFAULT) -> float:
    """Convierte True Airspeed (TAS) a Equivalent Airspeed (EAS): V_EAS = V_TAS * sqrt(rho / rho0)."""
    _validate_numeric(tas, density, rho0)
    if tas < 0:
        raise ValueError("TAS must be greater than or equal to zero")
    if density <= 0 or rho0 <= 0:
        raise ValueError("Density must be greater than zero")

    return tas * math.sqrt(density / rho0)


def eas_to_tas(eas: float, density: float, rho0: float = RHO0_DEFAULT) -> float:
    """Convierte Equivalent Airspeed (EAS) a True Airspeed (TAS): V_TAS = V_EAS * sqrt(rho0 / rho)."""
    _validate_numeric(eas, density, rho0)
    if eas < 0:
        raise ValueError("EAS must be greater than or equal to zero")
    if density <= 0 or rho0 <= 0:
        raise ValueError("Density must be greater than zero")

    return eas * math.sqrt(rho0 / density)


def tas_to_mach(tas: float, speed_of_sound: float) -> float:
    """Convierte True Airspeed (TAS) a número de Mach: M = V_TAS / a."""
    _validate_numeric(tas, speed_of_sound)
    if tas < 0:
        raise ValueError("TAS must be greater than or equal to zero")
    if speed_of_sound <= 0:
        raise ValueError("Speed of sound must be greater than zero")

    return tas / speed_of_sound


def mach_to_tas(mach: float, speed_of_sound: float) -> float:
    """Convierte número de Mach a True Airspeed (TAS): V_TAS = M * a."""
    _validate_numeric(mach, speed_of_sound)
    if mach < 0:
        raise ValueError("Mach must be greater than or equal to zero")
    if speed_of_sound <= 0:
        raise ValueError("Speed of sound must be greater than zero")

    return mach * speed_of_sound


# =============================================================================
# PRESIÓN DE IMPACTO Y CALIBRATED AIRSPEED (SUB/SUPERSÓNICA)
# =============================================================================

def cas_to_impact_pressure(
    cas: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Calcula la presión de impacto qc a partir de CAS (soporta regímenes subsónico y supersónico)."""
    _validate_numeric(cas, p0, rho0, gamma)
    if cas < 0:
        raise ValueError("CAS must be greater than or equal to zero")
    if p0 <= 0 or rho0 <= 0:
        raise ValueError("Reference pressure and density must be greater than zero")
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")

    if cas == 0.0:
        return 0.0

    a0 = math.sqrt(gamma * p0 / rho0)
    mc = cas / a0

    if mc <= 1.0:
        # Fórmula de Saint-Venant (subsónica)
        term = 1.0 + ((gamma - 1.0) / 2.0) * (mc**2)
        return p0 * ((term ** (gamma / (gamma - 1.0))) - 1.0)
    else:
        # Fórmula de Rayleigh-Pitot supersónica
        term1 = ((gamma + 1.0) / 2.0) * (mc**2)
        term2 = ((gamma + 1.0) ** 2 * (mc**2)) / (4.0 * gamma * (mc**2) - 2.0 * (gamma - 1.0))
        return p0 * (term1 * (term2 ** (1.0 / (gamma - 1.0))) - 1.0)


def impact_pressure_to_cas(
    qc: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Calcula Calibrated Airspeed (CAS) a partir de la presión de impacto qc (subsónica y supersónica)."""
    _validate_numeric(qc, p0, rho0, gamma)
    if qc < 0:
        raise ValueError("Impact pressure must be greater than or equal to zero")
    if p0 <= 0 or rho0 <= 0:
        raise ValueError("Reference pressure and density must be greater than zero")
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")

    if qc == 0.0:
        return 0.0

    a0 = math.sqrt(gamma * p0 / rho0)
    qc_ratio_sonic = (((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))) - 1.0

    if (qc / p0) <= qc_ratio_sonic:
        # Inversión analítica exacta de Saint-Venant
        term = ((qc / p0) + 1.0) ** ((gamma - 1.0) / gamma) - 1.0
        mc = math.sqrt((2.0 / (gamma - 1.0)) * term)
    else:
        # Inversión numérica de Rayleigh-Pitot
        def residual(m):
            term1 = ((gamma + 1.0) / 2.0) * (m**2)
            term2 = ((gamma + 1.0) ** 2 * (m**2)) / (4.0 * gamma * (m**2) - 2.0 * (gamma - 1.0))
            return (term1 * (term2 ** (1.0 / (gamma - 1.0))) - 1.0) - (qc / p0)

        mc = solve_brent(residual, 1.0, 50.0, tolerance=1e-10)

    return mc * a0


def cas_to_mach(
    cas: float,
    pressure: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Calcula el número de Mach de vuelo a partir de CAS y la presión estática ambiente."""
    _validate_numeric(cas, pressure, p0, rho0, gamma)
    if cas < 0:
        raise ValueError("CAS must be greater than or equal to zero")
    if pressure <= 0 or p0 <= 0 or rho0 <= 0:
        raise ValueError("Pressures and densities must be greater than zero")
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")

    if cas == 0.0:
        return 0.0

    qc = cas_to_impact_pressure(cas, p0=p0, rho0=rho0, gamma=gamma)
    qc_p = qc / pressure
    sonic_limit = (((gamma + 1.0) / 2.0) ** (gamma / (gamma - 1.0))) - 1.0

    if qc_p <= sonic_limit:
        term = (qc_p + 1.0) ** ((gamma - 1.0) / gamma) - 1.0
        return math.sqrt(max(0.0, (2.0 / (gamma - 1.0)) * term))
    else:
        # Flujo supersónico en el tubo pitot
        def residual(m):
            term1 = ((gamma + 1.0) / 2.0) * (m**2)
            term2 = ((gamma + 1.0) ** 2 * (m**2)) / (4.0 * gamma * (m**2) - 2.0 * (gamma - 1.0))
            return (term1 * (term2 ** (1.0 / (gamma - 1.0))) - 1.0) - qc_p

        return solve_brent(residual, 1.0, 50.0, tolerance=1e-10)


def mach_to_cas(
    mach: float,
    pressure: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Calcula Calibrated Airspeed (CAS) a partir del número de Mach y la presión estática ambiente."""
    _validate_numeric(mach, pressure, p0, rho0, gamma)
    if mach < 0:
        raise ValueError("Mach must be greater than or equal to zero")
    if pressure <= 0 or p0 <= 0 or rho0 <= 0:
        raise ValueError("Pressures and densities must be greater than zero")
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")

    if mach == 0.0:
        return 0.0

    if mach <= 1.0:
        total_p_ratio = (1.0 + ((gamma - 1.0) / 2.0) * (mach**2)) ** (gamma / (gamma - 1.0))
        qc = pressure * (total_p_ratio - 1.0)
    else:
        term1 = ((gamma + 1.0) / 2.0) * (mach**2)
        term2 = ((gamma + 1.0) ** 2 * (mach**2)) / (4.0 * gamma * (mach**2) - 2.0 * (gamma - 1.0))
        p02_p1 = term1 * (term2 ** (1.0 / (gamma - 1.0)))
        qc = pressure * (p02_p1 - 1.0)

    return impact_pressure_to_cas(qc, p0=p0, rho0=rho0, gamma=gamma)


def cas_to_tas(
    cas: float,
    pressure: float,
    speed_of_sound: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Convierte Calibrated Airspeed (CAS) a True Airspeed (TAS)."""
    mach = cas_to_mach(cas, pressure, p0=p0, rho0=rho0, gamma=gamma)
    return mach * speed_of_sound


def tas_to_cas(
    tas: float,
    pressure: float,
    speed_of_sound: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Convierte True Airspeed (TAS) a Calibrated Airspeed (CAS)."""
    mach = tas_to_mach(tas, speed_of_sound)
    return mach_to_cas(mach, pressure, p0=p0, rho0=rho0, gamma=gamma)


def cas_to_eas(
    cas: float,
    pressure: float,
    density: float,
    speed_of_sound: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Convierte Calibrated Airspeed (CAS) a Equivalent Airspeed (EAS)."""
    tas = cas_to_tas(cas, pressure, speed_of_sound, p0=p0, rho0=rho0, gamma=gamma)
    return tas_to_eas(tas, density, rho0=rho0)


def eas_to_cas(
    eas: float,
    pressure: float,
    density: float,
    speed_of_sound: float,
    p0: float = P0_DEFAULT,
    rho0: float = RHO0_DEFAULT,
    gamma: float = GAMMA_DEFAULT,
) -> float:
    """Convierte Equivalent Airspeed (EAS) a Calibrated Airspeed (CAS)."""
    tas = eas_to_tas(eas, density, rho0=rho0)
    return tas_to_cas(tas, pressure, speed_of_sound, p0=p0, rho0=rho0, gamma=gamma)
