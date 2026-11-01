# Changelog

Todas las modificaciones notables a PyADinic están documentadas en este archivo.

## [0.3.0] - 2026-11-01

### Añadido
- **Atmósfera ISA con desviación térmica:**
  - Soporte de parámetro `delta_T` en `isa_model(altitude, is_geometric=True, delta_T=0.0)`. La altitud se interpreta como altitud de presión estándar.
  - Función `density_altitude(pressure_altitude, delta_T=0.0, is_geometric=True)` para calcular la altitud de densidad utilizando el algoritmo de Brent.
- **Perfiles aerodinámicos avanzados:**
  - Soporte para la serie NACA modificada de 4 y 5 dígitos (sufijo `-ab`) con distribución de espesor adaptada en dos tramos (`thickness_distribution_modified`).
  - Parámetro `thickness_func` en `naca4` y `naca5` para inyectar distribuciones arbitrarias de espesor.
  - Generación de perfiles a partir de coordenadas arbitrarias con `airfoil_from_coordinates(name, x, y)`.
  - Carga y normalización de perfiles en formato Selig .dat con `airfoil_from_selig(source)`.
  - Helper `monotone_surface(x, y)` para garantizar coordenadas x estrictamente crecientes descartando duplicados cerca del borde de ataque.
  - Soporte de perfiles NACA modificados en la teoría de perfiles delgados (`thin_airfoil_theory`, `thin_airfoil_naca4`, `thin_airfoil_naca5`).
- **Geometría alar 3D:**
  - Dataclass `Wing` (inmutable).
  - Generación de geometría trapezoidal `trapezoidal_wing(span, root_chord, tip_chord, sweep_c4=0.0, deg=True)`.
  - Generación desde área y alargamiento `wing_from_area_ar(area, aspect_ratio, taper_ratio=1.0, sweep_c4=0.0, deg=True)`.
  - Cálculo de cuerda local `chord_at(wing, y)` y carga alar `wing_loading(weight, wing)`.
- **Paneles y Polares:**
  - Dataclass `PolarState` retornada por `panel_polar`, compatible con desempaquetado de tupla `(alpha, cl, cd, cm_c4)`.
  - `panel_analysis` y `panel_polar` ahora aceptan códigos NACA directamente como cadenas (`str`) de 4 o 5 dígitos, incluyendo perfiles modificados.
- **Pruebas y Validación:**
  - Suite de integración completa `test_integracion_mision.py` cubriendo atmósfera, aerodinámica 2D/3D, compresión y planeo con RK4.
  - Suite de pruebas unitarias para perfiles avanzados y formato Selig en `tests/test_foil.py`.
  - Suite de pruebas unitarias para geometría alar 3D en `tests/test_wing.py`.

### Modificado
- **Métodos numéricos de EDO:**
  - `rk4`, `euler` y `heun` garantizan que el punto final `tf` se alcance exactamente sin duplicar muestras por error de redondeo flotante.
- **Interpolación vectorizada:**
  - `linear_interpolation`, `lagrange_interpolation`, `newton_interpolation`, `cubic_spline_interpolation` y `pchip_interpolation` ahora aceptan tanto escalares como arreglos/listas de consulta `x`, devolviendo escalares `float` o `ndarray` correspondientemente.
- **Compatibilidad y obsolescencia:**
  - El parámetro `n_panels` de `panel_analysis` emite `DeprecationWarning` cuando se especifica un valor diferente a `None`.
