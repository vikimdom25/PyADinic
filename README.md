# PyADinic

[![Python](https://img.shields.io/badge/python-3.9%20%7C%203.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![Version](https://img.shields.io/badge/version-0.3.0-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**PyADinic** es una librería científica en Python diseñada para ingeniería aeroespacial, aerodinámica, dinámica de gases compresibles, diseño de perfiles y alas, y soporte computacional mediante métodos numéricos avanzados implementados desde primeros principios (solo depende de NumPy).

---

## Características Principales

### 1. Atmósfera Estándar Internacional (ISA / U.S. Standard 1976) y Desviaciones
- Modelo continuo de 7 capas desde nivel del mar hasta la mesopausa (0 a 84,852 m geopotenciales / 86,000 m geométricos).
- Soporte para desviaciones de temperatura atmosférica no estándar `delta_T` (altitud de presión estándar).
- Cálculo robusto de **altitud de densidad** (`density_altitude`) resolviendo la inversión con el método de Brent.
- Propiedades termodinámicas: temperatura ($T$), presión ($P$), densidad ($\rho$) y velocidad del sonido ($a$) vía `AtmosphericState`.

### 2. Geometría y Análisis de Perfiles Aerodinámicos
- **Generación de perfiles NACA:**
  - Serie NACA de 4 dígitos (`naca4`).
  - Serie NACA de 5 dígitos (`naca5`, perfiles normales y reflejos).
  - Serie NACA modificada de 4 y 5 dígitos con sufijo `-ab` (`thickness_distribution_modified`).
  - Distribuciones de espesor personalizables mediante callables (`thickness_func`).
- **Perfiles arbitrarios y formato Selig:**
  - Construcción de perfiles arbitrarios desde arreglos de coordenadas (`airfoil_from_coordinates`).
  - Carga y normalización de archivos `.dat` en formato Selig (`airfoil_from_selig`).
  - Acondicionamiento de superficies monótonas (`monotone_surface`).
- **Teoría de Perfiles Delgados (Thin Airfoil Theory):**
  - Coeficientes de Fourier ($A_0, A_1, A_2$).
  - Ángulo de sustentación nula ($\alpha_0$), pendiente de sustentación $dC_l/d\alpha$, momento en $c/4$ y centro de presiones.
- **Método de Paneles 2D (Hess-Smith):**
  - Fuentes constantes por panel con vórtice uniforme y condición de Kutta.
  - Distribución de presiones $C_p(x)$ suave y fuerzas aerodinámicas ($C_l, C_d, C_m$).
  - Acepta objetos `Airfoil` o códigos NACA directamente como cadenas (`"2412"`, `"23015"`).
  - Cálculo de polar aerodinámica vía `panel_polar` retornando `PolarState` (compatible con desempaquetado de tupla).

### 3. Geometría de Ala Tridimensional (3D)
- Dataclass inmutable `Wing` con parámetros completos: superficie ($S$), envergadura ($b$), alargamiento ($AR$), estrechamiento ($\lambda$), cuerda media aerodinámica ($MAC$), posición espanwise ($y_{\text{MAC}}$) y posición del borde de ataque ($x_{\text{LE,MAC}}$).
- Generación paramétrica con `trapezoidal_wing` y `wing_from_area_ar`.
- Cuerda local a lo largo de la envergadura `chord_at(wing, y)` y carga alar `wing_loading(weight, wing)`.

### 4. Dinámica de Gases, Choques y Expansión
- **Flujo Isentrópico:** Ratios $P_0/P$, $T_0/T$, $\rho_0/\rho$, presión de impacto $q_c$, relación de áreas $A/A^*$ subsónica y supersónica.
- **Ondas de Choque Normales y Oblicuas:** Relaciones de salto, pérdida de presión de estancamiento, relación de Rayleigh-Pitot, ecuación $\theta$-$\beta$-Mach, ángulo máximo de deflexión y ramas débil/fuerte.
- **Expansión de Prandtl-Meyer:** Función $\nu(M)$, ángulo de Mach $\mu$ y abanico de expansión.

### 5. Viscosidad y Conversión de Velocidades
- **Ley de Sutherland:** Viscosidad del aire y modelo genérico.
- **Velocidades Aeronáuticas:** Conversiones bidireccionales entre IAS, CAS, EAS, TAS y Mach.

### 6. Métodos Numéricos Integrados
- Solución de ecuaciones no lineales: Brent, Newton-Raphson, Secante, Bisección, Thomas.
- Interpolación: PCHIP, Splines Cúbicos, Newton, Lagrange, Lineal (soporte escalar y vectorial).
- Integración: Cuadratura Gauss-Legendre, Simpson 1/3, Simpson 3/8, Trapecio.
- EDO: RK4, Heun, Euler con control estricto de intervalos y tolerancia de endpoints.
- Álgebra lineal: LU, Cholesky, Gauss-Jordan, Eliminación Gaussiana.

---

## Instalación

Clona el repositorio e instala en modo editable:

```bash
git clone https://github.com/tu-usuario/PyADinic.git
cd PyADinic
pip install -e .
```

---

## Ejemplos de Uso

### 1. Atmósfera con Desviación ISA y Altitud de Densidad
```python
import pyadinic as pad

# ISA a 3,048 m (10,000 ft) con desviación ISA+20 °C
atm = pad.isa_model(3048, delta_T=20.0)
print(f"Temperatura: {atm.temperature:.2f} K")
print(f"Densidad:    {atm.density:.4f} kg/m³")

# Altitud de densidad correspondiente
h_dens = pad.density_altitude(3048, delta_T=20.0)
print(f"Altitud de densidad: {h_dens:.1f} m")
```

### 2. Generación de Perfiles NACA y Perfiles Modificados
```python
import pyadinic as pad

# NACA 2412 estándar
foil_std = pad.naca4("2412", n_points=100)

# NACA 0012 modificado (radio LE índice 6, espesor máx al 40% de cuerda)
foil_mod = pad.naca4("0012-64", n_points=100)

# Carga de perfil en formato Selig .dat
# foil_selig = pad.airfoil_from_selig("e387.dat")
```

### 3. Teoría de Perfiles Delgados (Thin Airfoil Theory)
```python
import pyadinic as pad

# Análisis analítico de un NACA 23015 a alpha = 4°
res = pad.thin_airfoil_naca5("23015", alpha=4.0, deg=True)
print(f"Cl (perfiles delgados): {res.cl:.4f}")
print(f"alpha_0:                {res.alpha_0:.4f} rad")
print(f"Cm_c/4:                 {res.cm_c4:.4f}")
```

### 4. Método de Paneles 2D (Hess-Smith) y Polares
```python
import pyadinic as pad

# Análisis a un ángulo de ataque pasando directamente el código NACA
state = pad.panel_analysis("2412", alpha=4.0, deg=True)
print(f"Cl: {state.cl:.4f}, Cd: {state.cd:.5f}, Cm_c4: {state.cm_c4:.4f}")

# Barrido polar
polar = pad.panel_polar("2412", [0.0, 2.0, 4.0, 6.0], deg=True)
print(f"Cl para cada ángulo: {polar.cl}")

# Compatible con desempaquetado de tupla tradicional:
alphas, cls, cds, cms = pad.panel_polar("2412", [0.0, 4.0], deg=True)
```

### 5. Geometría Alar 3D y Carga Alar
```python
import pyadinic as pad

# Definición de un ala trapezoidal (envergadura 12 m, raíz 2 m, punta 1 m)
wing = pad.trapezoidal_wing(span=12.0, root_chord=2.0, tip_chord=1.0, sweep_c4=15.0, deg=True)
print(f"Superficie alar: {wing.area:.2f} m²")
print(f"Aspect Ratio:    {wing.aspect_ratio:.2f}")
print(f"Cuerda MAC:      {wing.mac:.3f} m")

# Cuerda local en la mitad de la semienvergadura (y = 3 m)
c_local = pad.chord_at(wing, y=3.0)
print(f"Cuerda a y=3m:   {c_local:.3f} m")

# Carga alar para un peso de 25,000 N
ws = pad.wing_loading(weight=25000.0, wing=wing)
print(f"Carga alar:      {ws:.1f} N/m²")
```

---

## Pruebas

Para correr la suite completa de pruebas:

```bash
pytest
```

---

## Licencia

Distribuido bajo la Licencia MIT. Consulta el archivo `LICENSE` para más información.
