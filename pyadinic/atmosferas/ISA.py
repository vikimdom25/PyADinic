import math
import numpy as np
from pyadinic.core.dataclass import AtmosphericState
from pyadinic.core.constants import (
    R_e,
    g0,
    R,
    gamma_air,
    T0,
    P0,
    ISA_H_BOUNDS,
    ISA_LAPSE_RATES,
    MAX_GEOMETRIC_ALTITUDE,
    MAX_GEOPOTENTIAL_ALTITUDE,
)

# Precomputar estados base en los límites de cada capa estándar para máxima eficiencia y continuidad exacta
_LAYER_BASE_T = [T0]
_LAYER_BASE_P = [P0]

for _i in range(len(ISA_LAPSE_RATES)):
    _dh = ISA_H_BOUNDS[_i + 1] - ISA_H_BOUNDS[_i]
    _lapse = ISA_LAPSE_RATES[_i]
    if _lapse == 0.0:
        _t_next = _LAYER_BASE_T[-1]
        _p_next = _LAYER_BASE_P[-1] * math.exp(-g0 * _dh / (R * _t_next))
    else:
        _t_next = _LAYER_BASE_T[-1] + (_lapse * _dh)
        _p_next = _LAYER_BASE_P[-1] * ((_t_next / _LAYER_BASE_T[-1]) ** (-g0 / (R * _lapse)))
    _LAYER_BASE_T.append(_t_next)
    _LAYER_BASE_P.append(_p_next)


def isa_model(
    altitude: float,
    is_geometric: bool = True,
    delta_T: float = 0.0,
) -> AtmosphericState:
    """Calcula las propiedades termodinámicas de la Atmósfera Estándar Internacional (ISA / U.S. 1976)
    con soporte opcional para desviaciones de temperatura atmosférica (delta_T).

    Interpretación de altitud:
        En aviación e ingeniería aeronáutica, cuando la atmósfera no es estándar (delta_T != 0),
        la altitud se interpreta como ALTITUD DE PRESIÓN (pressure altitude). La presión estática
        del aire corresponde exactamente a la de la ISA estándar a dicha altitud, mientras que la
        temperatura real del aire es T = T_ISA + delta_T. La densidad (rho = p / (R * T)) y la
        velocidad del sonido (a = sqrt(gamma * R * T)) se derivan a partir de dicha presión y
        temperatura real mediante la ecuación del gas ideal y la termodinámica de gases.

    Args:
        altitude: Altitud en metros [m] (interpretada como altitud de presión).
        is_geometric: Si es True, interpreta la altitud como geométrica Z y la convierte a geopotencial H.
                      Si es False, interpreta directamente como altitud geopotencial H.
        delta_T: Desviación térmica respecto a la ISA estándar en Kelvin o Celsius [K o °C]. Por defecto 0.0.

    Returns:
        AtmosphericState con:
            - temperature [K]
            - pressure [Pa]
            - density [kg/m³]
            - speed_of_sound [m/s]

    Raises:
        TypeError: Si la altitud o delta_T no son numéricos, o si is_geometric no es booleano.
        ValueError: Si la altitud está fuera del rango soportado [0, 86000 m geométricas o 84852 m geopotenciales],
                    o si la temperatura resultante (T_ISA + delta_T) es menor o igual a cero.
    """
    # 1. Validaciones de Tipo
    if isinstance(altitude, bool) or not isinstance(altitude, (int, float, np.integer, np.floating)):
        raise TypeError(f"La altitud debe ser numérica, se recibió {type(altitude).__name__}")
    if isinstance(delta_T, bool) or not isinstance(delta_T, (int, float, np.integer, np.floating)):
        raise TypeError(f"delta_T debe ser numérico, se recibió {type(delta_T).__name__}")
    if not isinstance(is_geometric, bool):
        raise TypeError("is_geometric debe ser de tipo booleano")

    # 2. Conversión y Validaciones de Rango
    if is_geometric:
        if altitude < 0 or altitude > MAX_GEOMETRIC_ALTITUDE:
            raise ValueError(
                f"Altitud geométrica fuera de rango. El modelo soporta de 0 a {MAX_GEOMETRIC_ALTITUDE:,.0f} metros."
            )
        H = (R_e * float(altitude)) / (R_e + float(altitude))
    else:
        if altitude < 0 or altitude > MAX_GEOPOTENTIAL_ALTITUDE:
            raise ValueError(
                f"Altitud geopotencial fuera de rango. El modelo soporta de 0 a {MAX_GEOPOTENTIAL_ALTITUDE:,.0f} metros."
            )
        H = float(altitude)

    # 3. Identificación de la capa atmosférica (0 a 6)
    layer_idx = 0
    for idx in range(len(ISA_LAPSE_RATES)):
        if H >= ISA_H_BOUNDS[idx]:
            layer_idx = idx

    h_base = ISA_H_BOUNDS[layer_idx]
    t_base = _LAYER_BASE_T[layer_idx]
    p_base = _LAYER_BASE_P[layer_idx]
    lapse = ISA_LAPSE_RATES[layer_idx]
    delta_h = H - h_base

    # 4. Cálculo de Temperatura ISA y Presión según el gradiente térmico
    if lapse == 0.0:
        t_isa = t_base
        pressure = p_base * math.exp((-g0 / (R * t_base)) * delta_h)
    else:
        t_isa = t_base + (lapse * delta_h)
        pressure = p_base * ((t_isa / t_base) ** (-g0 / (R * lapse)))

    # Aplicar desviación de temperatura delta_T
    temperature = t_isa + float(delta_T)
    if temperature <= 0.0:
        raise ValueError(
            f"La temperatura resultante ({temperature:.2f} K) debe ser estrictamente positiva."
        )

    # 5. Cálculos Derivados (Ley de gas ideal y velocidad del sonido)
    density = pressure / (R * temperature)
    speed_of_sound = math.sqrt(gamma_air * R * temperature)

    return AtmosphericState(
        temperature=temperature,
        pressure=pressure,
        density=density,
        speed_of_sound=speed_of_sound,
    )


def density_altitude(
    pressure_altitude: float,
    delta_T: float = 0.0,
    is_geometric: bool = True,
) -> float:
    """Calcula la altitud de densidad a partir de la altitud de presión y la desviación térmica delta_T.

    La altitud de densidad es la altitud en la atmósfera estándar internacional (ISA, con delta_T = 0)
    a la cual la densidad del aire es idéntica a la densidad observada.

    Args:
        pressure_altitude: Altitud de presión en metros [m].
        delta_T: Desviación de temperatura respecto a la ISA en Kelvin o Celsius [K o °C]. Por defecto 0.0.
        is_geometric: Si es True, tanto la entrada (pressure_altitude) como la salida se expresan
                      en altitud geométrica [m]. Si es False, se interpretan y devuelven en
                      altitud geopotencial [m].

    Returns:
        Altitud de densidad en metros [m] (geométrica o geopotencial según is_geometric).

    Raises:
        TypeError: Si pressure_altitude o delta_T no son numéricos, o si is_geometric no es booleano.
        ValueError: Si la altitud de presión o la altitud de densidad calculada caen fuera del rango
                    soportado por el modelo ISA [0, 86000 m geométricas o 84852 m geopotenciales],
                    o si la temperatura resultante es no positiva.
    """
    if isinstance(pressure_altitude, bool) or not isinstance(pressure_altitude, (int, float, np.integer, np.floating)):
        raise TypeError(f"pressure_altitude debe ser numérica, se recibió {type(pressure_altitude).__name__}")
    if isinstance(delta_T, bool) or not isinstance(delta_T, (int, float, np.integer, np.floating)):
        raise TypeError(f"delta_T debe ser numérico, se recibió {type(delta_T).__name__}")
    if not isinstance(is_geometric, bool):
        raise TypeError("is_geometric debe ser booleano")

    if float(delta_T) == 0.0:
        isa_model(pressure_altitude, is_geometric=is_geometric, delta_T=0.0)
        return float(pressure_altitude)

    target_state = isa_model(pressure_altitude, is_geometric=is_geometric, delta_T=delta_T)
    target_rho = target_state.density

    from pyadinic.numerical_methods.solvers import solve_brent

    max_alt = MAX_GEOMETRIC_ALTITUDE if is_geometric else MAX_GEOPOTENTIAL_ALTITUDE
    rho_min = isa_model(max_alt, is_geometric=is_geometric, delta_T=0.0).density
    rho_max = isa_model(0.0, is_geometric=is_geometric, delta_T=0.0).density

    if target_rho < rho_min or target_rho > rho_max:
        raise ValueError(
            f"La altitud de densidad calculada está fuera del rango del modelo ISA [0, {max_alt:,.0f} m]."
        )

    def obj_func(h):
        return isa_model(h, is_geometric=is_geometric, delta_T=0.0).density - target_rho

    h_da = solve_brent(obj_func, 0.0, max_alt)
    return float(h_da)
