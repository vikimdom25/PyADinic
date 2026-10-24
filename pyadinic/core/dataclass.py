from dataclasses import dataclass
import numpy as np

@dataclass
class AtmosphericState:
    temperature: float
    pressure: float
    density: float
    speed_of_sound: float


@dataclass
class NormalShockState:
    mach_1: float
    mach_2: float
    pressure_ratio: float
    temperature_ratio: float
    density_ratio: float
    total_pressure_ratio: float
    pitot_pressure_ratio: float


@dataclass
class ObliqueShockState:
    mach_1: float
    mach_2: float
    wave_angle: float
    deflection_angle: float
    normal_mach_1: float
    normal_mach_2: float
    pressure_ratio: float
    temperature_ratio: float
    density_ratio: float
    total_pressure_ratio: float


@dataclass
class PrandtlMeyerState:
    mach_1: float
    mach_2: float
    turn_angle: float
    nu_1: float
    nu_2: float
    mu_1: float
    mu_2: float
    pressure_ratio: float
    temperature_ratio: float
    density_ratio: float

@dataclass
class Airfoil:
    name: str
    x_upper: np.ndarray
    y_upper: np.ndarray
    x_lower: np.ndarray
    y_lower: np.ndarray


@dataclass
class ThinAirfoilState:
    alpha: float
    cl: float
    cm_c4: float
    cm_le: float
    alpha_0: float
    lift_slope: float
    center_of_pressure: float
    a0: float
    a1: float
    a2: float


@dataclass
class PanelState:
    """
    Resultados del método de paneles 2D para un perfil aerodinámico.

    Atributos
    ---------
    alpha : float
        Ángulo de ataque en radianes.
    cl : float
        Coeficiente de sustentación 2D (ejes viento).
    cd : float
        Coeficiente de resistencia al avance 2D (≈0 en flujo potencial invíscido).
    cm_le : float
        Coeficiente de momento de cabeceo respecto al borde de ataque.
    cm_c4 : float
        Coeficiente de momento de cabeceo respecto al cuarto de cuerda.
    cn : float
        Coeficiente de fuerza normal (eje cuerpo).
    ca : float
        Coeficiente de fuerza axial (eje cuerpo).
    center_of_pressure : float
        Posición adimensional del centro de presiones x_cp / c.
    n_panels : int
        Número de paneles utilizados en la discretización.
    xc : np.ndarray
        Coordenadas x de los puntos de control de cada panel.
    yc : np.ndarray
        Coordenadas y de los puntos de control de cada panel.
    cp : np.ndarray
        Coeficiente de presión Cp en cada punto de control.
    gamma : float
        Intensidad del vórtice uniforme de Hess-Smith (positiva en sentido
        horario, normalizada a V_inf = 1). Cl ≈ 2·gamma·perímetro / c.
    sigma : np.ndarray
        Intensidades de fuente sigma_j de cada panel (normalizadas a V_inf = 1).
    """
    alpha: float
    cl: float
    cd: float
    cm_le: float
    cm_c4: float
    cn: float
    ca: float
    center_of_pressure: float
    n_panels: int
    xc: np.ndarray
    yc: np.ndarray
    cp: np.ndarray
    gamma: float
    sigma: np.ndarray



@dataclass
class PolarState:
    """
    Resultados de la polar aerodinamica calculada con el metodo de paneles.

    Atributos
    ---------
    alpha : np.ndarray
        Angulos de ataque en radianes.
    cl : np.ndarray
        Coeficientes de sustentacion Cl(alpha).
    cd : np.ndarray
        Coeficientes de resistencia al avance Cd(alpha).
    cm_c4 : np.ndarray
        Coeficientes de momento de cabeceo respecto al cuarto de cuerda Cm_c4(alpha).
    """
    alpha: np.ndarray
    cl: np.ndarray
    cd: np.ndarray
    cm_c4: np.ndarray

    def __iter__(self):
        return iter((self.alpha, self.cl, self.cd, self.cm_c4))
