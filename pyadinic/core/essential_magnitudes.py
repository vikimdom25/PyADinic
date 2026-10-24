import math
from .constants import R, gamma_air

def dynamic_pressure(rho, velocity):
    if not isinstance(rho, (int, float)) or not isinstance(velocity, (int, float)):
        raise TypeError("Error, no numeric args")
    if isinstance(rho, bool) or isinstance(velocity, bool):
        raise TypeError("Error, no numeric args")
    
    if rho <= 0 or velocity < 0:
        raise ValueError("Error, magnitudes below or equal zero")
    
    dp = 0.5 * rho * (velocity**2)  
    return dp

def mach_number(air_speed, speed_of_sound):
    if not isinstance(air_speed, (int, float)) or not isinstance(speed_of_sound, (int, float)):
        raise TypeError("Error, no numeric args")
    if isinstance(air_speed, bool) or isinstance(speed_of_sound, bool):
        raise TypeError("Error, no numeric args")
    if air_speed < 0 or speed_of_sound < 0:
        raise ValueError("Error, magnitudes below zero")
    if speed_of_sound == 0:
        raise ValueError("Speed of sound must be greater than zero")

    Ma = air_speed / speed_of_sound
    return Ma

def reynolds_number(air_speed, characteristic_length, density, dynamic_viscosity):
    magnitudes = [air_speed, characteristic_length, density, dynamic_viscosity]
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int, float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
        if magnitude <= 0:
            raise ValueError("Error, magnitudes below or equal zero")    

    Re = (air_speed * characteristic_length * density) / dynamic_viscosity
    return Re

def speed_of_sound(temperature, gamma=gamma_air, R=R):
    magnitudes = [temperature, gamma, R]

    for magnitude in magnitudes:
        if not isinstance(magnitude, (int, float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
        if magnitude < 0:
            raise ValueError("Negative values not allowed")
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")
    
    sos = math.sqrt(gamma * R * temperature)
    return sos


def pressure_coefficient(pressure, freestream_pressure, dynamic_pressure_inf):
    """
    Calcula el coeficiente de presión: Cp = (p - p_inf) / q_inf.
    """
    if isinstance(dynamic_pressure_inf, bool) or not isinstance(dynamic_pressure_inf, (int, float)):
        raise TypeError("dynamic_pressure_inf must be numeric")
    if dynamic_pressure_inf <= 0:
        raise ValueError("Dynamic pressure must be positive")

    if isinstance(freestream_pressure, bool) or not isinstance(freestream_pressure, (int, float)):
        raise TypeError("freestream_pressure must be numeric")

    import numpy as np
    if isinstance(pressure, (np.ndarray, list, tuple)):
        p = np.asarray(pressure, dtype=float)
        return (p - freestream_pressure) / dynamic_pressure_inf

    if isinstance(pressure, bool) or not isinstance(pressure, (int, float)):
        raise TypeError("pressure must be numeric")

    return (pressure - freestream_pressure) / dynamic_pressure_inf


def pressure_coefficient_incompressible(velocity, freestream_velocity):
    """
    Calcula el coeficiente de presión en flujo incompresible: Cp = 1 - (V / V_inf)^2.
    """
    if isinstance(freestream_velocity, bool) or not isinstance(freestream_velocity, (int, float)):
        raise TypeError("freestream_velocity must be numeric")
    if freestream_velocity <= 0:
        raise ValueError("Freestream velocity must be positive")

    import numpy as np
    if isinstance(velocity, (np.ndarray, list, tuple)):
        v = np.asarray(velocity, dtype=float)
        if np.any(v < 0):
            raise ValueError("Velocity cannot be negative")
        return 1.0 - (v / freestream_velocity) ** 2

    if isinstance(velocity, bool) or not isinstance(velocity, (int, float)):
        raise TypeError("velocity must be numeric")
    if velocity < 0:
        raise ValueError("Velocity cannot be negative")

    return 1.0 - (velocity / freestream_velocity) ** 2


def lift_force(cl, dynamic_pressure_inf, reference_area):
    """
    Calcula la fuerza de sustentación L = Cl * q * S.
    """
    for val, name in [(cl, "cl"), (dynamic_pressure_inf, "dynamic_pressure_inf"), (reference_area, "reference_area")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if dynamic_pressure_inf < 0:
        raise ValueError("Dynamic pressure cannot be negative")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")

    return cl * dynamic_pressure_inf * reference_area


def lift_coefficient(lift, dynamic_pressure_inf, reference_area):
    """
    Calcula el coeficiente de sustentación Cl = L / (q * S).
    """
    for val, name in [(lift, "lift"), (dynamic_pressure_inf, "dynamic_pressure_inf"), (reference_area, "reference_area")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if dynamic_pressure_inf <= 0:
        raise ValueError("Dynamic pressure must be positive")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")

    return lift / (dynamic_pressure_inf * reference_area)


def drag_force(cd, dynamic_pressure_inf, reference_area):
    """
    Calcula la fuerza de resistencia aerodinámica D = Cd * q * S.
    """
    for val, name in [(cd, "cd"), (dynamic_pressure_inf, "dynamic_pressure_inf"), (reference_area, "reference_area")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if cd < 0:
        raise ValueError("Drag coefficient cannot be negative")
    if dynamic_pressure_inf < 0:
        raise ValueError("Dynamic pressure cannot be negative")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")

    return cd * dynamic_pressure_inf * reference_area


def drag_coefficient(drag, dynamic_pressure_inf, reference_area):
    """
    Calcula el coeficiente de resistencia Cd = D / (q * S).
    """
    for val, name in [(drag, "drag"), (dynamic_pressure_inf, "dynamic_pressure_inf"), (reference_area, "reference_area")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if drag < 0:
        raise ValueError("Drag force cannot be negative")
    if dynamic_pressure_inf <= 0:
        raise ValueError("Dynamic pressure must be positive")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")

    return drag / (dynamic_pressure_inf * reference_area)


def pitching_moment(cm, dynamic_pressure_inf, reference_area, reference_length):
    """
    Calcula el momento de cabeceo M = Cm * q * S * c.
    """
    for val, name in [(cm, "cm"), (dynamic_pressure_inf, "dynamic_pressure_inf"), 
                      (reference_area, "reference_area"), (reference_length, "reference_length")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if dynamic_pressure_inf < 0:
        raise ValueError("Dynamic pressure cannot be negative")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")
    if reference_length <= 0:
        raise ValueError("Reference length must be positive")

    return cm * dynamic_pressure_inf * reference_area * reference_length


def pitching_moment_coefficient(moment, dynamic_pressure_inf, reference_area, reference_length):
    """
    Calcula el coeficiente de momento de cabeceo Cm = M / (q * S * c).
    """
    for val, name in [(moment, "moment"), (dynamic_pressure_inf, "dynamic_pressure_inf"), 
                      (reference_area, "reference_area"), (reference_length, "reference_length")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if dynamic_pressure_inf <= 0:
        raise ValueError("Dynamic pressure must be positive")
    if reference_area <= 0:
        raise ValueError("Reference area must be positive")
    if reference_length <= 0:
        raise ValueError("Reference length must be positive")

    return moment / (dynamic_pressure_inf * reference_area * reference_length)


def lift_to_drag_ratio(cl, cd):
    """
    Calcula la eficiencia aerodinámica L / D = Cl / Cd.
    """
    for val, name in [(cl, "cl"), (cd, "cd")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if cd <= 0:
        raise ValueError("Drag coefficient must be strictly positive")

    return cl / cd


def body_to_wind_coefficients(cn, ca, alpha, deg=False):
    """
    Convierte coeficientes en ejes cuerpo (CN, CA) a ejes viento (CL, CD):
    CL = CN * cos(alpha) - CA * sin(alpha)
    CD = CN * sin(alpha) + CA * cos(alpha)
    """
    for val, name in [(cn, "cn"), (ca, "ca"), (alpha, "alpha")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    a = math.radians(alpha) if deg else float(alpha)

    cos_a = math.cos(a)
    sin_a = math.sin(a)

    cl = cn * cos_a - ca * sin_a
    cd = cn * sin_a + ca * cos_a

    return cl, cd


def wind_to_body_coefficients(cl, cd, alpha, deg=False):
    """
    Convierte coeficientes en ejes viento (CL, CD) a ejes cuerpo (CN, CA):
    CN = CL * cos(alpha) + CD * sin(alpha)
    CA = -CL * sin(alpha) + CD * cos(alpha)
    """
    for val, name in [(cl, "cl"), (cd, "cd"), (alpha, "alpha")]:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be numeric")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    a = math.radians(alpha) if deg else float(alpha)

    cos_a = math.cos(a)
    sin_a = math.sin(a)

    cn = cl * cos_a + cd * sin_a
    ca = -cl * sin_a + cd * cos_a

    return cn, ca


def skin_friction_coefficient_laminar(reynolds):
    """
    Coeficiente de fricción medio en placa plana laminar (solución de Blasius):
    Cf = 1.328 / sqrt(Re).
    """
    if isinstance(reynolds, bool) or not isinstance(reynolds, (int, float)):
        raise TypeError("reynolds must be numeric")
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive")

    return 1.328 / math.sqrt(reynolds)


def skin_friction_coefficient_turbulent(reynolds, model="prandtl"):
    """
    Coeficiente de fricción medio en placa plana turbulenta.
    Modelos:
      - 'prandtl': Cf = 0.074 / Re^(1/5)
      - 'schlichting': Cf = 0.455 / (log10(Re))^2.58
    """
    if isinstance(reynolds, bool) or not isinstance(reynolds, (int, float)):
        raise TypeError("reynolds must be numeric")
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive")

    if model == "prandtl":
        return 0.074 / (reynolds ** 0.2)
    elif model == "schlichting":
        return 0.455 / (math.log10(reynolds) ** 2.58)
    else:
        raise ValueError("Unknown model: choose 'prandtl' or 'schlichting'")


def skin_friction_coefficient(reynolds, regime="auto", re_critical=5e5):
    """
    Calcula el coeficiente de fricción promedio en placa plana según el régimen:
    - 'laminar': solución de Blasius.
    - 'turbulent': modelo de Prandtl.
    - 'auto': laminar si Re <= re_critical, o con corrección de transición.
    """
    if isinstance(reynolds, bool) or not isinstance(reynolds, (int, float)):
        raise TypeError("reynolds must be numeric")
    if reynolds <= 0:
        raise ValueError("Reynolds number must be positive")

    if isinstance(re_critical, bool) or not isinstance(re_critical, (int, float)):
        raise TypeError("re_critical must be numeric")
    if re_critical <= 0:
        raise ValueError("re_critical must be positive")

    if regime == "laminar":
        return skin_friction_coefficient_laminar(reynolds)
    elif regime == "turbulent":
        return skin_friction_coefficient_turbulent(reynolds)
    elif regime == "auto":
        if reynolds <= re_critical:
            return skin_friction_coefficient_laminar(reynolds)
        else:
            cf_turb_crit = 0.074 / (re_critical ** 0.2)
            cf_lam_crit = 1.328 / math.sqrt(re_critical)
            a_term = re_critical * (cf_turb_crit - cf_lam_crit)
            return (0.074 / (reynolds ** 0.2)) - (a_term / reynolds)
    else:
        raise ValueError("regime must be 'laminar', 'turbulent', or 'auto'")


def boundary_layer_thickness(x, reynolds_x, regime="laminar"):
    """
    Espesor de la capa límite delta sobre placa plana a una distancia x:
    - Laminar: delta = 5.0 * x / sqrt(Re_x)
    - Turbulento: delta = 0.37 * x / Re_x^0.2
    """
    if isinstance(x, bool) or not isinstance(x, (int, float)):
        raise TypeError("x must be numeric")
    if x < 0:
        raise ValueError("x cannot be negative")

    if x == 0:
        return 0.0

    if isinstance(reynolds_x, bool) or not isinstance(reynolds_x, (int, float)):
        raise TypeError("reynolds_x must be numeric")
    if reynolds_x <= 0:
        raise ValueError("Reynolds number must be positive")

    if regime == "laminar":
        return 5.0 * x / math.sqrt(reynolds_x)
    elif regime == "turbulent":
        return 0.37 * x / (reynolds_x ** 0.2)
    else:
        raise ValueError("regime must be 'laminar' or 'turbulent'")

