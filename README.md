# PyADinic

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Tests](https://img.shields.io/badge/tests-437%20passed-brightgreen.svg)](https://pytest.org/)
[![Version](https://img.shields.io/badge/version-0.2.0-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PyADinic** es una librería científica en Python diseñada para ingeniería aeroespacial, aerodinámica, dinámica de gases compresibles y soporte computacional mediante métodos numéricos avanzados implementados desde primeros principios.

---

## Características Principales

### 1. Atmósfera Estándar Internacional (ISA / U.S. Standard 1976)
- Modelo continuo de 7 capas desde nivel del mar hasta la mesopausa (0 a 84,852 m geopotenciales / 86,000 m geométricos):
  - Troposfera, Tropopausa, Estratosfera (inferior y superior), Estratopausa, Mesosfera (inferior y superior).
- Conversión exacta entre altitud geométrica ($Z$) y geopotencial ($H$).
- Propiedades termodinámicas: temperatura ($T$), presión ($P$), densidad ($\rho$) y velocidad del sonido ($a$).
- Retorno estructurado mediante `AtmosphericState`.

### 2. Magnitudes Esenciales de Aerodinámica
- **Presión Dinámica ($q$):** $q = \frac{1}{2}\rho V^2$.
- **Número de Mach ($M$):** $M = V / a$.
- **Número de Reynolds ($Re$):** $Re = \frac{\rho V L}{\mu}$.
- **Velocidad del Sonido ($a$):** $a = \sqrt{\gamma R T}$.

### 3. Dinámica de Gases y Flujo Isentrópico
- Relaciones de estancamiento: temperatura total ($T_0$), presión total ($P_0$), densidad total ($\rho_0$).
- Presión de impacto ($q_c$).
- Inversión de ratios $P_0/P$, $T_0/T$ y $\rho_0/\rho$ para despejar Mach.
- Relación de áreas compresible ($A/A^*$):
  - Cálculo directo $A/A^*(M)$.
  - Inversión robusta $M(A/A^*)$ para ramas **subsónica** y **supersónica** empleando el algoritmo de Brent.

### 4. Ondas de Choque Normales y Oblicuas
- **Choque Normal:**
  - Relaciones de salto de Mach ($M_2$), presión ($P_2/P_1$), densidad ($\rho_2/\rho_1$), temperatura ($T_2/T_1$).
  - Pérdida de presión de estancamiento ($P_{02}/P_{01}$).
  - Relación de Rayleigh-Pitot ($P_{02}/P_1$) e inversión de Mach supersónico a partir de mediciones Pitot.
  - Retorno estructurado con `NormalShockState`.
- **Choque Oblicuo:**
  - Ecuación $\theta$-$\beta$-Mach exacta.
  - Cálculo analítico del ángulo máximo de deflexión ($\theta_{\max}$) y ángulo de choque asociado ($\beta_{\max}$).
  - Resolución de ramas de choque débil (*weak shock*) y fuerte (*strong shock*).
  - Detección de choque desprendido (*detached / bow shock*).
  - Retorno estructurado con `ObliqueShockState`.

### 5. Abanico de Expansión de Prandtl-Meyer
- Ángulo de Mach ($\mu = \arcsin(1/M)$).
- Función de Prandtl-Meyer $\nu(M)$ y ángulo máximo teórico de giro ($\nu_{\max}$).
- Inversión exacta $M(\nu)$.
- Expansión isentrópica completa tras un giro $\theta$ con relaciones de presión, temperatura y densidad (`PrandtlMeyerState`).

### 6. Viscosidad y Conversión de Velocidades Aeronáuticas
- **Ley de Sutherland:** Modelo para aire y formulación genérica de 3 constantes.
- **Conversiones Aeronáuticas Integrales:**
  - Indicated Airspeed (IAS) $\leftrightarrow$ Calibrated Airspeed (CAS) con correcciones de posición e instrumento.
  - Equivalent Airspeed (EAS) $\leftrightarrow$ True Airspeed (TAS): $V_{\text{EAS}} = V_{\text{TAS}} \sqrt{\rho / \rho_0}$.
  - Calibrated Airspeed (CAS) $\leftrightarrow$ Mach y TAS (soporta regímenes subsónico y supersónico vía Saint-Venant y Rayleigh-Pitot).
  - CAS $\leftrightarrow$ EAS.

### 7. Métodos Numéricos Robustos (Sin dependencias opacas)
- **Solución de Ecuaciones No Lineales:**
  - Método de Brent (combinación garantizada de interpolación inversa cuadrática, secante y bisección).
  - Newton-Raphson (con derivación numérica central).
  - Método de la Secante y Bisección clásica.
  - Algoritmo de Thomas para sistemas tridiagonales $O(N)$.
- **Interpolación Numérica:**
  - PCHIP (*Piecewise Cubic Hermite Interpolating Polynomial*) que preserva monotonicidad sin sobreoscilaciones.
  - Splines Cúbicos Naturales.
  - Polinomios de Newton (diferencias divididas) y Lagrange.
  - Interpolación lineal acotada.
- **Integración Numérica:**
  - Cuadratura Gauss-Legendre (cálculo dinámico de raíces de Legendre vía Newton).
  - Reglas de Simpson 1/3, Simpson 3/8 y Trapecio compuesto.
- **Ecuaciones Diferenciales Ordinarias (EDO):**
  - Runge-Kutta de 4to orden (RK4) con soporte para sistemas vectoriales.
  - Método de Heun (RK2) y Euler explícito.
- **Sistemas de Ecuaciones Lineales:**
  - Eliminación Gaussiana con pivoteo parcial.
  - Gauss-Jordan.

---

## Instalación

Clona el repositorio e instala el paquete en modo editable:

```bash
git clone https://github.com/tu-usuario/PyADinic.git
cd PyADinic
pip install -e .
```

O instala las dependencias de desarrollo para ejecutar la suite de pruebas:

```bash
pip install -e .[dev]
```

---

## Ejemplos Rápidos de Uso

### 1. Atmósfera ISA hasta 84,000 m
```python
import pyadinic as pad

# Nivel de crucero (11,000 m)
tropopausa = pad.isa_model(11000)
print(f"Tropopausa: T = {tropopausa.temperature:.2f} K, P = {tropopausa.pressure:.1f} Pa")

# Mesosfera alta (80,000 m)
mesosfera = pad.isa_model(80000)
print(f"Mesosfera:  T = {mesosfera.temperature:.2f} K, P = {mesosfera.pressure:.3f} Pa")
```

### 2. Ondas de Choque Normales y Supersónicas
```python
import pyadinic as pad

# Choque normal a Mach 2.0
choque = pad.normal_shock_relations(mach1=2.0)
print(f"Mach aguas abajo M2: {choque.mach_2:.4f}")
print(f"Salto de presión P2/P1: {choque.pressure_ratio:.2f}")
print(f"Pérdida de presión total P02/P01: {choque.total_pressure_ratio:.4f}")

# Despejar Mach de vuelo a partir de presión Pitot supersónica
p02_p1 = 5.6404  # P02 medido por Pitot / P1 ambiente
m1 = pad.mach1_from_rayleigh_pitot(p02_p1)
print(f"Mach calculado desde Pitot: {m1:.2f}")
```

### 3. Ondas de Choque Oblicuas (Ecuación θ-β-Mach)
```python
import pyadinic as pad

# Mach 2.0 sobre una cuña con deflexión theta = 10 grados
theta_max, beta_max = pad.max_deflection_angle(mach1=2.0, use_degrees=True)
print(f"Deflexión máxima antes de desprendimiento: {theta_max:.2f}°")

# Onda de choque oblicua débil
choque_oblicuo = pad.oblique_shock_relations(mach1=2.0, theta=10.0, branch="weak", use_degrees=True)
print(f"Ángulo de choque beta: {choque_oblicuo.wave_angle:.2f}°")
print(f"Mach aguas abajo M2:    {choque_oblicuo.mach_2:.4f}")
print(f"Salto de presión P2/P1: {choque_oblicuo.pressure_ratio:.2f}")
```

### 4. Expansión de Prandtl-Meyer
```python
import pyadinic as pad

# Flujo a M1 = 1.5 que gira una esquina con deflexión theta = 10 grados
expansion = pad.prandtl_meyer_expansion(mach1=1.5, theta=10.0, use_degrees=True)
print(f"Mach expandido M2: {expansion.mach_2:.4f}")
print(f"P2 / P1 tras expansión: {expansion.pressure_ratio:.4f}")
```

### 5. Conversiones de Velocidad Aeronáutica
```python
import pyadinic as pad

cas = 180.0  # m/s
presion_altitud = 26500.0  # Pa (~10 km)
a = 299.53  # m/s

# CAS a Mach y True Airspeed
mach = pad.cas_to_mach(cas, pressure=presion_altitud)
tas = pad.cas_to_tas(cas, pressure=presion_altitud, speed_of_sound=a)
print(f"Mach de vuelo: {mach:.4f}")
print(f"TAS:           {tas:.2f} m/s")
```

---

## Estructura del Proyecto

```
PyADinic/
├── pyproject.toml              # Configuración moderna de empaquetado (PEP 517/621)
├── .gitignore                  # Reglas de exclusión para Git
├── README.md                   # Documentación principal
├── LICENSE                     # Licencia MIT
├── pyadinic/                   # Paquete principal
│   ├── __init__.py             # Fachada de exportación unificada del API
│   ├── atmosferas/
│   │   ├── __init__.py
│   │   └── ISA.py              # Atmósfera ISA completa (7 capas hasta 86 km)
│   ├── core/
│   │   ├── __init__.py
│   │   ├── constants.py        # Constantes físicas y tablas de capas ISA
│   │   ├── dataclass.py        # Dataclasses (AtmosphericState, Shocks, Expansion)
│   │   └── essential_magnitudes.py # Magnitudes aerodinámicas fundamentales
│   ├── modelos/
│   │   ├── __init__.py
│   │   ├── isentropic.py       # Dinámica de gases y relaciones isentrópicas
│   │   ├── sutherland.py       # Modelo de viscosidad dinámica de Sutherland
│   │   ├── velocidades.py      # Conversiones completas IAS, CAS, EAS, TAS, Mach
│   │   ├── shocks.py           # Ondas de choque normales y oblicuas
│   │   └── prandtl_meyer.py    # Abanico de expansión de Prandtl-Meyer
│   └── numerical_methods/
│       ├── __init__.py
│       ├── differentiation.py  # Diferencias finitas centrales
│       ├── integration.py      # Cuadratura Gauss-Legendre, Simpson, Trapecio
│       ├── interpolation.py    # PCHIP, Splines Cúbicos, Newton, Lagrange
│       ├── linear_systems.py   # Eliminación Gaussiana, Gauss-Jordan
│       ├── ode.py              # Runge-Kutta 4, Heun, Euler
│       └── solvers.py          # Brent, Newton, Secante, Bisección, Thomas
└── tests/                      # Suite de 437 pruebas unitarias automatizadas
    ├── test_init.py
    ├── test_isa.py
    ├── test_isentropic.py
    ├── test_shocks.py
    ├── test_prandtl_meyer.py
    ├── test_velocidades.py
    ├── numerical_test.py
    └── ...
```

---

## Ejecución de Pruebas

Para ejecutar la suite completa de pruebas unitarias:

```bash
pytest
```

---

## Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
