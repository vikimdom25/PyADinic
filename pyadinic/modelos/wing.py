import math
from typing import Union
import numpy as np

from pyadinic.core.dataclass import Wing


def trapezoidal_wing(
    span: float,
    root_chord: float,
    tip_chord: float,
    sweep_c4: float = 0.0,
    deg: bool = True,
) -> Wing:
    """
    Construye la geometria de un ala trapezoidal recta o en flecha.

    Parametros
    ----------
    span : float
        Envergadura total alar b en metros [m].
    root_chord : float
        Cuerda en el encastre o raiz c_r en metros [m].
    tip_chord : float
        Cuerda en la punta c_t en metros [m].
    sweep_c4 : float, optional
        Angulo de flecha en el cuarto de cuerda (c/4). Por defecto 0.0.
    deg : bool, optional
        Si True, sweep_c4 se interpreta en grados. Si False, en radianes. Por defecto True.

    Retorna
    -------
    Wing
        Dataclass inmutable con las propiedades geometricas del ala.
    """
    if isinstance(span, (bool, np.bool_)) or not isinstance(span, (int, float, np.integer, np.floating)):
        raise TypeError(f"span debe ser numerico, se recibio {type(span).__name__}")
    if isinstance(root_chord, (bool, np.bool_)) or not isinstance(root_chord, (int, float, np.integer, np.floating)):
        raise TypeError(f"root_chord debe ser numerico, se recibio {type(root_chord).__name__}")
    if isinstance(tip_chord, (bool, np.bool_)) or not isinstance(tip_chord, (int, float, np.integer, np.floating)):
        raise TypeError(f"tip_chord debe ser numerico, se recibio {type(tip_chord).__name__}")
    if isinstance(sweep_c4, (bool, np.bool_)) or not isinstance(sweep_c4, (int, float, np.integer, np.floating)):
        raise TypeError(f"sweep_c4 debe ser numerico, se recibio {type(sweep_c4).__name__}")
    if not isinstance(deg, bool):
        raise TypeError(f"deg debe ser booleano, se recibio {type(deg).__name__}")

    b = float(span)
    cr = float(root_chord)
    ct = float(tip_chord)

    if b <= 0.0:
        raise ValueError("span debe ser estrictamente positivo")
    if cr <= 0.0:
        raise ValueError("root_chord debe ser estrictamente positivo")
    if ct < 0.0:
        raise ValueError("tip_chord debe ser no negativo")

    area = b * (cr + ct) / 2.0
    aspect_ratio = (b ** 2) / area
    taper_ratio = ct / cr
    mac = (2.0 / 3.0) * cr * (1.0 + taper_ratio + taper_ratio ** 2) / (1.0 + taper_ratio)
    sweep_rad = math.radians(sweep_c4) if deg else float(sweep_c4)

    return Wing(
        span=b,
        root_chord=cr,
        tip_chord=ct,
        area=area,
        aspect_ratio=aspect_ratio,
        taper_ratio=taper_ratio,
        mac=mac,
        sweep_c4=sweep_rad,
    )


def wing_from_area_ar(
    area: float,
    aspect_ratio: float,
    taper_ratio: float = 1.0,
    sweep_c4: float = 0.0,
    deg: bool = True,
) -> Wing:
    """
    Construye la geometria de un ala trapezoidal a partir de su superficie alar
    y su alargamiento (aspect ratio).

    Parametros
    ----------
    area : float
        Superficie alar de referencia S en metros cuadrados [m²].
    aspect_ratio : float
        Alargamiento o relacion de aspecto AR = b² / S.
    taper_ratio : float, optional
        Estrechamiento o conicidad lambda = c_t / c_r. Por defecto 1.0 (ala recta).
    sweep_c4 : float, optional
        Angulo de flecha en cuarto de cuerda. Por defecto 0.0.
    deg : bool, optional
        Si True, sweep_c4 se interpreta en grados. Por defecto True.

    Retorna
    -------
    Wing
        Dataclass inmutable con las propiedades geometricas del ala.
    """
    if isinstance(area, (bool, np.bool_)) or not isinstance(area, (int, float, np.integer, np.floating)):
        raise TypeError(f"area debe ser numerica, se recibio {type(area).__name__}")
    if isinstance(aspect_ratio, (bool, np.bool_)) or not isinstance(aspect_ratio, (int, float, np.integer, np.floating)):
        raise TypeError(f"aspect_ratio debe ser numerico, se recibio {type(aspect_ratio).__name__}")
    if isinstance(taper_ratio, (bool, np.bool_)) or not isinstance(taper_ratio, (int, float, np.integer, np.floating)):
        raise TypeError(f"taper_ratio debe ser numerico, se recibio {type(taper_ratio).__name__}")
    if isinstance(sweep_c4, (bool, np.bool_)) or not isinstance(sweep_c4, (int, float, np.integer, np.floating)):
        raise TypeError(f"sweep_c4 debe ser numerico, se recibio {type(sweep_c4).__name__}")
    if not isinstance(deg, bool):
        raise TypeError(f"deg debe ser booleano, se recibio {type(deg).__name__}")

    s = float(area)
    ar = float(aspect_ratio)
    lam = float(taper_ratio)

    if s <= 0.0:
        raise ValueError("area debe ser estrictamente positiva")
    if ar <= 0.0:
        raise ValueError("aspect_ratio debe ser estrictamente positivo")
    if lam < 0.0:
        raise ValueError("taper_ratio debe ser no negativo")

    span = math.sqrt(s * ar)
    root_chord = 2.0 * s / (span * (1.0 + lam))
    tip_chord = root_chord * lam
    mac = (2.0 / 3.0) * root_chord * (1.0 + lam + lam ** 2) / (1.0 + lam)
    sweep_rad = math.radians(sweep_c4) if deg else float(sweep_c4)

    return Wing(
        span=span,
        root_chord=root_chord,
        tip_chord=tip_chord,
        area=s,
        aspect_ratio=ar,
        taper_ratio=lam,
        mac=mac,
        sweep_c4=sweep_rad,
    )


def chord_at(wing: Wing, y: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Calcula la cuerda geometrica local c(y) a una estacion de semienvergadura y.

    Parametros
    ----------
    wing : Wing
        Geometria del ala.
    y : float o array-like
        Posicion envergadura en metros [m] respecto al plano de simetria (y=0).
        Debe cumplir |y| <= span / 2.

    Retorna
    -------
    float o ndarray
        Cuerda local c(y) en metros [m].
    """
    if not isinstance(wing, Wing):
        raise TypeError(f"wing debe ser una instancia de Wing, se recibio {type(wing).__name__}")

    b_half = wing.span / 2.0

    def _eval_single(y_val):
        if isinstance(y_val, (bool, np.bool_)) or not isinstance(y_val, (int, float, np.integer, np.floating)):
            raise TypeError("y debe ser numerico")
        y_f = float(y_val)
        if abs(y_f) > b_half + 1e-9:
            raise ValueError(f"La estacion y={y_f} m esta fuera de la semienvergadura [-{b_half}, {b_half}] m")
        eta = min(1.0, abs(y_f) / b_half)
        return wing.root_chord + eta * (wing.tip_chord - wing.root_chord)

    if np.isscalar(y):
        return _eval_single(y)
    else:
        y_arr = np.asarray(y)
        return np.vectorize(_eval_single, otypes=[float])(y_arr)


def wing_loading(weight: float, wing: Union[Wing, float]) -> float:
    """
    Calcula la carga alar W / S [N/m²] para un peso dado y una superficie alar.

    Parametros
    ----------
    weight : float
        Peso total de la aeronave en Newtons [N].
    wing : Wing o float
        Instancia de Wing o valor directo de la superficie alar S en [m²].

    Retorna
    -------
    float
        Carga alar en N/m².
    """
    if isinstance(weight, (bool, np.bool_)) or not isinstance(weight, (int, float, np.integer, np.floating)):
        raise TypeError(f"weight debe ser numerico, se recibio {type(weight).__name__}")

    w = float(weight)
    if w <= 0.0:
        raise ValueError("weight debe ser estrictamente positivo")

    if isinstance(wing, Wing):
        s = wing.area
    elif isinstance(wing, (int, float, np.integer, np.floating)) and not isinstance(wing, bool):
        s = float(wing)
    else:
        raise TypeError(f"wing debe ser una instancia de Wing o un area numerica, se recibio {type(wing).__name__}")

    if s <= 0.0:
        raise ValueError("La superficie alar debe ser estrictamente positiva")

    return float(w / s)
