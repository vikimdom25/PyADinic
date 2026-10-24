# Progreso de PyADinic v0.3.0

## Parte A: Limpieza
- [x] A1: Corrección de paso y duplicados en ODE (`rk4`, `euler`, `heun`) en `numerical_methods/ode.py`
- [x] A2: Vectorización de interpolación en `numerical_methods/interpolation.py`
- [x] A3: Firmas flexibles en `panel_methods.py`, dataclass `PolarState` y DeprecationWarning para `n_panels`
- [x] A4: Helper `monotone_surface` en `modelos/naca_airfoils.py`
- [ ] A5: Migración de `mision_pyadinic.py` a `tests/test_integracion_mision.py`
- [ ] A6: Documentación en `README.md`, versión `0.3.0` en `pyproject.toml`, `CHANGELOG.md` y actualización de `__version__`

## Parte B: Huecos Funcionales
- [ ] B1: Atmósfera con desviación `delta_T` y `density_altitude` en `atmosferas/ISA.py`
- [ ] B2: Perfiles NACA modificados (`-ab`), `thickness_func`, `airfoil_from_coordinates` y `airfoil_from_selig` en `modelos/naca_airfoils.py`
- [ ] B3: Geometría de ala `Wing` en `core/dataclass.py` y `modelos/wing.py`
