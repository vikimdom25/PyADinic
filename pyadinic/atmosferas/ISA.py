import math
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


def isa_model(altitude: float, is_geometric: bool = True) -> AtmosphericState:
    """Calcula las propiedades termodinámicas de la Atmósfera Estándar Internacional (ISA / U.S. 1976).

    Soporta altitudes desde nivel del mar (0 m) hasta la mesopausa (84,852 m geopotenciales / 86,000 m geométricos).

    Args:
        altitude: Altitud en metros [m].
        is_geometric: Si es True, interpreta la altitud como geométrica Z y la convierte a geopotencial H.
                      Si es False, interpreta directamente como altitud geopotencial H.

    Returns:
        AtmosphericState con:
            - temperature [K]
            - pressure [Pa]
            - density [kg/m³]
            - speed_of_sound [m/s]

    Raises:
        TypeError: Si la altitud no es numérica o si is_geometric no es booleano.
        ValueError: Si la altitud está fuera del rango soportado [0, 86000 m geométricas o 84852 m geopotenciales].
    """
    # 1. Validaciones de Tipo
    if isinstance(altitude, bool) or not isinstance(altitude, (int, float)):
        raise TypeError(f"La altitud debe ser numérica, se recibió {type(altitude).__name__}")
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

    # 4. Cálculo de Temperatura y Presión según el gradiente térmico
    if lapse == 0.0:
        temperature = t_base
        pressure = p_base * math.exp((-g0 / (R * t_base)) * delta_h)
    else:
        temperature = t_base + (lapse * delta_h)
        pressure = p_base * ((temperature / t_base) ** (-g0 / (R * lapse)))

    # 5. Cálculos Derivados (Ley de gas ideal y velocidad del sonido)
    density = pressure / (R * temperature)
    speed_of_sound = math.sqrt(gamma_air * R * temperature)

    return AtmosphericState(
        temperature=temperature,
        pressure=pressure,
        density=density,
        speed_of_sound=speed_of_sound,
    )
