"""
Método de Paneles 2D — Hess-Smith (fuentes constantes por panel + un vórtice
uniforme sobre todo el contorno), con condición de Kutta en el borde de salida.

Implementación basada en las herramientas propias de PyADinic:
  - Geometría del perfil: módulo naca_airfoils (Airfoil, naca4, naca5).
  - Resolución del sistema lineal [A]{x} = {b}: lu_solve de linear_systems.

Formulación
-----------
Incógnitas: sigma_1..sigma_N (intensidad de fuente de cada panel) y gamma
(intensidad de vórtice, común a todos los paneles).  Ecuaciones:

  * N condiciones de no penetración (velocidad normal nula) en los puntos
    medios de los paneles.
  * 1 condición de Kutta: velocidades tangenciales de salida iguales y de
    sentido opuesto en el primer y el último panel,  Vt_1 + Vt_N = 0.

Resultado: sistema cuadrado (N+1)x(N+1).  Se eligió Hess-Smith en lugar de
vórtice constante por panel porque esta última produce un zigzag panel a panel
en Cp(x) (modo par/impar mal condicionado), mientras que Hess-Smith da una
distribución de presión suave.

Convención de orientación del contorno
---------------------------------------
Sentido **antihorario** (área de Shoelace positiva): borde de salida →
extradós (hacia el LE) → borde de ataque → intradós → borde de salida.  El
interior queda a la izquierda del tangente y la normal EXTERIOR a la DERECHA:
    nx_ext = ty ,  ny_ext = -tx

gamma es positiva en sentido HORARIO (Katz & Plotkin); con esta convención
gamma > 0 corresponde a Cl > 0 y  Cl ≈ 2·gamma·(perímetro)/c.

Todas las magnitudes están normalizadas con V_inf = 1 y cuerda c = 1.

Referencias
-----------
Hess, J.L. & Smith, A.M.O., "Calculation of potential flow about arbitrary
bodies", Progress in Aeronautical Sciences, 8, 1–138, 1967.

Katz, J. & Plotkin, A., "Low-Speed Aerodynamics", 2nd ed., Cambridge Univ.
Press, 2001. Capítulos 10–11.
"""

import math
import numpy as np

from pyadinic.core.dataclass import Airfoil, PanelState
from pyadinic.numerical_methods.linear_systems import lu_solve
from .naca_airfoils import naca4 as _naca4, naca5 as _naca5


# ---------------------------------------------------------------------------
# 1. Geometría de paneles
# ---------------------------------------------------------------------------

def panel_geometry(airfoil):
    """
    Construye el contorno de paneles en sentido ANTIHORARIO a partir de un Airfoil.

    Orden resultante:
      TE → extradós (de TE a LE) → LE → intradós (de LE a TE) → TE

    Parámetros
    ----------
    airfoil : Airfoil
        Perfil generado por naca4/naca5.  Los arrays x_upper, y_upper, x_lower,
        y_lower van de borde de ataque (índice 0) a borde de salida (índice -1).

    Retorna
    -------
    xp : ndarray, shape (N+1,)
        Coordenadas x de los N+1 nodos del contorno cerrado.
    yp : ndarray, shape (N+1,)
        Coordenadas y de los N+1 nodos del contorno cerrado.
    """
    if not isinstance(airfoil, Airfoil):
        raise TypeError("airfoil must be an Airfoil instance")

    x_upper = np.asarray(airfoil.x_upper, dtype=float)
    y_upper = np.asarray(airfoil.y_upper, dtype=float)
    x_lower = np.asarray(airfoil.x_lower, dtype=float)
    y_lower = np.asarray(airfoil.y_lower, dtype=float)

    if x_upper.ndim != 1 or len(x_upper) < 2:
        raise ValueError("Airfoil coordinate arrays must be 1D with at least 2 points")

    # Sentido antihorario: TE → extradós invertido → LE → intradós → TE
    # x_upper[0] ≈ 0 (LE), x_upper[-1] ≈ 1 (TE)
    x_upper_rev = x_upper[::-1]   # TE → LE
    y_upper_rev = y_upper[::-1]

    # Concatenar: TE→LE por extradós (sin duplicar LE) + LE→TE por intradós
    xp = np.concatenate([x_upper_rev, x_lower[1:]])
    yp = np.concatenate([y_upper_rev, y_lower[1:]])

    return xp, yp


def _panel_properties(xp, yp):
    """
    Calcula la geometría de los N paneles (puntos de control, normales, tangentes).

    Para contorno antihorario, la normal EXTERIOR apunta a la DERECHA del tangente:
        nx = ty,  ny = -tx

    Parámetros
    ----------
    xp, yp : ndarray, shape (N+1,)

    Retorna
    -------
    xc, yc : ndarray, shape (N,)   — puntos de control (mitad de cada panel)
    nx, ny : ndarray, shape (N,)   — normal exterior unitaria
    tx, ty : ndarray, shape (N,)   — tangente unitaria (dirección del panel)
    length : ndarray, shape (N,)   — longitud de cada panel
    """
    dx = xp[1:] - xp[:-1]
    dy = yp[1:] - yp[:-1]
    length = np.sqrt(dx**2 + dy**2)

    tx = dx / length
    ty = dy / length

    # Contorno antihorario → normal exterior gira 90° a la DERECHA del tangente
    nx = ty
    ny = -tx

    xc = 0.5 * (xp[1:] + xp[:-1])
    yc = 0.5 * (yp[1:] + yp[:-1])

    return xc, yc, nx, ny, tx, ty, length


# ---------------------------------------------------------------------------
# 2. Coeficientes de influencia (Katz & Plotkin, ec. 11.1 – 11.14)
# ---------------------------------------------------------------------------

def _local_frame(xp, yp):
    """Datos comunes de los paneles: puntos medios, cosenos/senos y vectores a extremos."""
    xc = 0.5 * (xp[1:] + xp[:-1])
    yc = 0.5 * (yp[1:] + yp[:-1])
    dx = xp[1:] - xp[:-1]
    dy = yp[1:] - yp[:-1]
    lj = np.hypot(dx, dy)
    cos_j = dx / lj
    sin_j = dy / lj

    r1x = xc[:, None] - xp[None, :-1]
    r1y = yc[:, None] - yp[None, :-1]
    r2x = xc[:, None] - xp[None, 1:]
    r2y = yc[:, None] - yp[None, 1:]

    s1 = r1x * cos_j[None, :] + r1y * sin_j[None, :]
    s2 = r2x * cos_j[None, :] + r2y * sin_j[None, :]
    n1 = -r1x * sin_j[None, :] + r1y * cos_j[None, :]

    with np.errstate(divide="ignore", invalid="ignore"):
        ln_r1_r2 = 0.5 * np.log((r1x**2 + r1y**2) / (r2x**2 + r2y**2))
    theta1 = np.arctan2(n1, s1)
    theta2 = np.arctan2(n1, s2)
    return cos_j, sin_j, ln_r1_r2, theta1, theta2


def _vortex_influence_matrices(xp, yp):
    """
    Velocidad inducida (U, V) en el punto de control de cada panel i por un
    vórtice de intensidad unitaria, positiva en sentido HORARIO, en el panel j.

    En coordenadas locales del panel j (s a lo largo, n a la izquierda):
        u_s = (1/2pi) * (theta2 - theta1)
        u_n = (1/2pi) * ln(r2 / r1)

    Autoinfluencia (i == j): el valor principal es nulo y el salto de velocidad
    tangencial a través del panel vale gamma; en la cara EXTERIOR (derecha del
    tangente) u_s = -1/2, u_n = 0.
    """
    cos_j, sin_j, ln_r1_r2, theta1, theta2 = _local_frame(xp, yp)
    inv_2pi = 1.0 / (2.0 * math.pi)
    u_s = inv_2pi * (theta2 - theta1)
    u_n = -inv_2pi * ln_r1_r2

    diag = np.arange(len(cos_j))
    u_s[diag, diag] = -0.5
    u_n[diag, diag] = 0.0

    U = u_s * cos_j[None, :] - u_n * sin_j[None, :]
    V = u_s * sin_j[None, :] + u_n * cos_j[None, :]
    return U, V


def _source_influence_matrices(xp, yp):
    """
    Velocidad inducida (U, V) en el punto de control de cada panel i por una
    fuente de intensidad unitaria en el panel j.

    En coordenadas locales del panel j:
        u_s = (1/2pi) * ln(r1 / r2)
        u_n = (1/2pi) * (theta2 - theta1)

    Autoinfluencia (i == j): u_s = 0 y, en la cara exterior (derecha del
    tangente), u_n = -1/2 respecto de la normal izquierda, es decir +1/2 a lo
    largo de la normal exterior.
    """
    cos_j, sin_j, ln_r1_r2, theta1, theta2 = _local_frame(xp, yp)
    inv_2pi = 1.0 / (2.0 * math.pi)
    u_s = inv_2pi * ln_r1_r2
    u_n = inv_2pi * (theta2 - theta1)

    diag = np.arange(len(cos_j))
    u_s[diag, diag] = 0.0
    u_n[diag, diag] = -0.5

    U = u_s * cos_j[None, :] - u_n * sin_j[None, :]
    V = u_s * sin_j[None, :] + u_n * cos_j[None, :]
    return U, V


def _tangential_and_normal(xp, yp):
    """Matrices de influencia proyectadas sobre normal y tangente de cada panel."""
    xc, yc, nx, ny, tx, ty, length = _panel_properties(xp, yp)
    Us, Vs = _source_influence_matrices(xp, yp)
    Uv, Vv = _vortex_influence_matrices(xp, yp)

    Sn = Us * nx[:, None] + Vs * ny[:, None]   # fuentes -> velocidad normal
    St = Us * tx[:, None] + Vs * ty[:, None]   # fuentes -> velocidad tangencial
    # El vórtice es uniforme: se suma la contribución de todos los paneles
    Vn = (Uv * nx[:, None] + Vv * ny[:, None]).sum(axis=1)
    Vt = (Uv * tx[:, None] + Vv * ty[:, None]).sum(axis=1)
    return (xc, yc, nx, ny, tx, ty, length), Sn, St, Vn, Vt


def influence_matrix(xp, yp, alpha_rad):
    """
    Ensambla la matriz de influencia A ((N+1)x(N+1)) y el vector b (N+1).

    Incógnitas x = [sigma_1, …, sigma_N, gamma].

      Filas i = 0..N-1 (no penetración):
        Σ_j sigma_j · Sn_ij  +  gamma · Vn_i  =  -(u_inf · nx_i + v_inf · ny_i)

      Fila N (Kutta, Vt_0 + Vt_{N-1} = 0):
        Σ_j sigma_j · (St_0j + St_{N-1,j})  +  gamma · (Vt_0 + Vt_{N-1})
            =  -(V_inf · (t_0 + t_{N-1}))

    Parámetros
    ----------
    xp, yp : ndarray, shape (N+1,)
        Nodos del contorno (salida de panel_geometry).
    alpha_rad : float
        Ángulo de ataque en radianes.

    Retorna
    -------
    A : ndarray, shape (N+1, N+1)
    b_vec : ndarray, shape (N+1,)
    xc, yc, nx, ny, tx, ty, length : ndarray, shape (N,)
    """
    geom, Sn, St, Vn, Vt = _tangential_and_normal(xp, yp)
    xc, yc, nx, ny, tx, ty, length = geom
    n = len(xc)

    u_inf = math.cos(alpha_rad)
    v_inf = math.sin(alpha_rad)

    A = np.zeros((n + 1, n + 1))
    b_vec = np.zeros(n + 1)

    A[:n, :n] = Sn
    A[:n, n] = Vn
    b_vec[:n] = -(u_inf * nx + v_inf * ny)

    A[n, :n] = St[0] + St[n - 1]
    A[n, n] = Vt[0] + Vt[n - 1]
    b_vec[n] = -(u_inf * (tx[0] + tx[n - 1]) + v_inf * (ty[0] + ty[n - 1]))

    return A, b_vec, xc, yc, nx, ny, tx, ty, length


# ---------------------------------------------------------------------------
# 3. Resolución del sistema lineal
# ---------------------------------------------------------------------------

def panel_solve(xp, yp, alpha_rad):
    """
    Resuelve el sistema lineal de Hess-Smith con lu_solve de PyADinic.

    Parámetros
    ----------
    xp, yp : ndarray, shape (N+1,)
        Nodos del contorno (salida de panel_geometry).
    alpha_rad : float
        Ángulo de ataque en radianes.

    Retorna
    -------
    sigma : ndarray, shape (N,)
        Intensidad de fuente de cada panel (normalizada con V_inf = 1).
    gamma : float
        Intensidad del vórtice uniforme (positiva en sentido horario).
    A, b_vec : sistema lineal ensamblado.
    xc, yc, nx, ny, tx, ty, length : ndarray, shape (N,)
    """
    A, b_vec, xc, yc, nx, ny, tx, ty, length = influence_matrix(xp, yp, alpha_rad)
    x = np.asarray(lu_solve(A, b_vec), dtype=float)
    sigma = x[:-1]
    gamma = float(x[-1])
    return sigma, gamma, A, b_vec, xc, yc, nx, ny, tx, ty, length


# ---------------------------------------------------------------------------
# 4. Distribución de presión
# ---------------------------------------------------------------------------

def panel_pressure(sigma, gamma, xp, yp, alpha_rad, tx, ty):
    """
    Calcula la velocidad tangencial y el coeficiente de presión en cada panel.

        V_t_i = (u_inf · tx_i + v_inf · ty_i) + Σ_j sigma_j · St_ij + gamma · Vt_i

    Coeficiente de presión (flujo incompresible, V_inf = 1):
        Cp_i = 1 − V_t_i²

    Parámetros
    ----------
    sigma : ndarray, shape (N,)
    gamma : float
    xp, yp : ndarray, shape (N+1,)
    alpha_rad : float
    tx, ty : ndarray, shape (N,)
        Tangentes unitarias de los paneles.

    Retorna
    -------
    cp : ndarray, shape (N,)
    vt : ndarray, shape (N,)
    """
    sigma = np.asarray(sigma, dtype=float)
    _, _, St, _, Vt = _tangential_and_normal(xp, yp)

    u_inf = math.cos(alpha_rad)
    v_inf = math.sin(alpha_rad)
    vt = (u_inf * tx + v_inf * ty) + St @ sigma + gamma * Vt
    cp = 1.0 - vt**2
    return cp, vt


# ---------------------------------------------------------------------------
# 5. Fuerzas aerodinámicas
# ---------------------------------------------------------------------------

def panel_forces(cp, xc, yc, nx, ny, length, alpha_rad):
    """
    Integra el coeficiente de presión para obtener los coeficientes aerodinámicos.

    La fuerza de presión por unidad de cuerda actúa hacia el interior del cuerpo
    con magnitud Cp * ds:
        Fn_i = −Cp_i * ny_i * length_i   (componente normal)
        Fa_i = −Cp_i * nx_i * length_i   (componente axial)

    Los coeficientes en ejes viento se obtienen rotando por el ángulo de ataque:
        Cl = Cn * cos(alpha) − Ca * sin(alpha)
        Cd = Cn * sin(alpha) + Ca * cos(alpha)

    El coeficiente de momento respecto al borde de ataque:
        Cm_le = Σ_i Cp_i * (xc_i * ny_i − yc_i * nx_i) * length_i

    Parámetros
    ----------
    cp : ndarray, shape (N,)
    xc, yc, nx, ny, length : ndarray, shape (N,)
    alpha_rad : float

    Retorna
    -------
    cl, cd, cm_le, cm_c4, cn, ca : float
    """
    # Fuerzas en ejes cuerpo (integración del coeficiente de presión sobre el contorno)
    cn = float(np.sum(-cp * ny * length))
    ca = float(np.sum(-cp * nx * length))

    # Momento respecto al borde de ataque (positivo en sentido cabeceo a nariz arriba)
    cm_le = float(np.sum(cp * (xc * ny - yc * nx) * length))

    # Transformación a ejes viento
    cos_a = math.cos(alpha_rad)
    sin_a = math.sin(alpha_rad)
    cl = cn * cos_a - ca * sin_a
    cd = cn * sin_a + ca * cos_a

    # Momento en cuarto de cuerda (la fuerza normal al eje x es la que tiene brazo)
    cm_c4 = cm_le + 0.25 * cn

    return cl, cd, cm_le, cm_c4, cn, ca


# ---------------------------------------------------------------------------
# 6. Análisis completo
# ---------------------------------------------------------------------------

def panel_analysis(airfoil, alpha, deg=False, n_panels=None):
    """
    Análisis aerodinámico completo de un perfil mediante el método de paneles 2D.

    Parámetros
    ----------
    airfoil : Airfoil
        Perfil aerodinámico (de naca4, naca5 o similar).
    alpha : float
        Ángulo de ataque en radianes (o grados si deg=True).
    deg : bool, optional
        Si True, alpha se interpreta en grados. Por defecto False.
    n_panels : int or None, optional
        No utilizado en esta versión (la discretización es la del Airfoil
        recibido: usa n_points en naca4/naca5). Reservado para compatibilidad.

    Retorna
    -------
    PanelState
        Objeto con todos los resultados del análisis.
    """
    if not isinstance(airfoil, Airfoil):
        raise TypeError("airfoil must be an Airfoil instance")

    if isinstance(alpha, bool) or not isinstance(alpha, (int, float)):
        raise TypeError("alpha must be numeric")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    alpha_rad = math.radians(alpha) if deg else float(alpha)

    xp, yp = panel_geometry(airfoil)

    n = len(xp) - 1
    if n < 4:
        raise ValueError("Airfoil must have at least 4 panels")

    sigma, gamma, A_mat, b_vec, xc, yc, nx, ny, tx, ty, length = panel_solve(xp, yp, alpha_rad)
    cp, vt = panel_pressure(sigma, gamma, xp, yp, alpha_rad, tx, ty)
    cl, cd, cm_le, cm_c4, cn, ca = panel_forces(cp, xc, yc, nx, ny, length, alpha_rad)

    if abs(cn) < 1e-12:
        x_cp = 0.25
    else:
        x_cp = -cm_le / cn

    return PanelState(
        alpha=alpha_rad,
        cl=cl,
        cd=cd,
        cm_le=cm_le,
        cm_c4=cm_c4,
        cn=cn,
        ca=ca,
        center_of_pressure=x_cp,
        n_panels=n,
        xc=xc,
        yc=yc,
        cp=cp,
        gamma=gamma,
        sigma=sigma,
    )


def panel_polar(airfoil, alpha_list, deg=False):
    """
    Calcula la polar aerodinámica (Cl, Cd, Cm) para una lista de ángulos de ataque.

    Parámetros
    ----------
    airfoil : Airfoil
        Perfil aerodinámico.
    alpha_list : list or array-like of float
        Lista de ángulos de ataque.
    deg : bool, optional
        Si True, los ángulos se interpretan en grados. Por defecto False.

    Retorna
    -------
    alphas : ndarray
        Ángulos de ataque en radianes.
    cls : ndarray
        Coeficientes de sustentación Cl(alpha).
    cds : ndarray
        Coeficientes de resistencia Cd(alpha) (≈0 en flujo potencial).
    cms : ndarray
        Coeficientes de momento en c/4.
    """
    if not isinstance(airfoil, Airfoil):
        raise TypeError("airfoil must be an Airfoil instance")

    try:
        alpha_arr = np.asarray(alpha_list, dtype=float)
    except (TypeError, ValueError):
        raise TypeError("alpha_list must be a list or array of numeric values")

    if alpha_arr.ndim != 1 or len(alpha_arr) < 1:
        raise ValueError("alpha_list must be a non-empty 1D array")

    if not isinstance(deg, bool):
        raise TypeError("deg must be a boolean")

    alphas_rad = np.radians(alpha_arr) if deg else alpha_arr.copy()
    cls = np.zeros(len(alphas_rad))
    cds = np.zeros(len(alphas_rad))
    cms = np.zeros(len(alphas_rad))

    for k, a in enumerate(alphas_rad):
        state = panel_analysis(airfoil, a, deg=False)
        cls[k] = state.cl
        cds[k] = state.cd
        cms[k] = state.cm_c4

    return alphas_rad, cls, cds, cms


# ---------------------------------------------------------------------------
# 7. Atajos directos por código NACA
# ---------------------------------------------------------------------------

def panel_naca4(code, alpha=0.0, deg=False, n_points=100):
    """
    Atajo directo para analizar un perfil NACA 4 dígitos con el método de paneles.

    Parámetros
    ----------
    code : str
        Código NACA de 4 dígitos (p. ej. '2412').
    alpha : float, optional
        Ángulo de ataque en radianes (o grados si deg=True). Por defecto 0.
    deg : bool, optional
        Si True, alpha se interpreta en grados. Por defecto False.
    n_points : int, optional
        Número de puntos de discretización del perfil. Por defecto 100.

    Retorna
    -------
    PanelState
    """
    airfoil = _naca4(code, n_points=n_points)
    return panel_analysis(airfoil, alpha=alpha, deg=deg)


def panel_naca5(code, alpha=0.0, deg=False, n_points=100):
    """
    Atajo directo para analizar un perfil NACA 5 dígitos con el método de paneles.

    Parámetros
    ----------
    code : str
        Código NACA de 5 dígitos (p. ej. '23012').
    alpha : float, optional
        Ángulo de ataque en radianes (o grados si deg=True). Por defecto 0.
    deg : bool, optional
        Si True, alpha se interpreta en grados. Por defecto False.
    n_points : int, optional
        Número de puntos de discretización del perfil. Por defecto 100.

    Retorna
    -------
    PanelState
    """
    airfoil = _naca5(code, n_points=n_points)
    return panel_analysis(airfoil, alpha=alpha, deg=deg)
