import math
import numpy as np
import pytest

from pyadinic.core.essential_magnitudes import (
    pressure_coefficient,
    pressure_coefficient_incompressible,
    lift_force,
    lift_coefficient,
    drag_force,
    drag_coefficient,
    pitching_moment,
    pitching_moment_coefficient,
    lift_to_drag_ratio,
    body_to_wind_coefficients,
    wind_to_body_coefficients,
    skin_friction_coefficient_laminar,
    skin_friction_coefficient_turbulent,
    skin_friction_coefficient,
    boundary_layer_thickness,
)


# ==============================================================================
# Pressure Coefficient
# ==============================================================================

def test_pressure_coefficient_scalar():
    p = 95000.0
    p_inf = 101325.0
    q_inf = 5000.0

    cp = pressure_coefficient(p, p_inf, q_inf)
    assert cp == pytest.approx((95000.0 - 101325.0) / 5000.0)


def test_pressure_coefficient_array():
    p = np.array([101325.0, 95000.0, 90000.0])
    p_inf = 101325.0
    q_inf = 5000.0

    cp = pressure_coefficient(p, p_inf, q_inf)
    assert isinstance(cp, np.ndarray)
    assert cp[0] == pytest.approx(0.0)
    assert cp[1] == pytest.approx((95000.0 - 101325.0) / 5000.0)


def test_pressure_coefficient_invalid_inputs():
    with pytest.raises(ValueError):
        pressure_coefficient(100000, 101325, 0)

    with pytest.raises(ValueError):
        pressure_coefficient(100000, 101325, -100)

    with pytest.raises(TypeError):
        pressure_coefficient("invalid", 101325, 5000)

    with pytest.raises(TypeError):
        pressure_coefficient(100000, True, 5000)

    with pytest.raises(TypeError):
        pressure_coefficient(100000, 101325, "bad")


def test_pressure_coefficient_incompressible_scalar():
    v = 50.0
    v_inf = 100.0

    cp = pressure_coefficient_incompressible(v, v_inf)
    assert cp == pytest.approx(1.0 - 0.25)


def test_pressure_coefficient_incompressible_stagnation():
    cp = pressure_coefficient_incompressible(0.0, 100.0)
    assert cp == pytest.approx(1.0)


def test_pressure_coefficient_incompressible_array():
    v = np.array([0.0, 50.0, 100.0, 150.0])
    cp = pressure_coefficient_incompressible(v, 100.0)

    assert cp[0] == pytest.approx(1.0)
    assert cp[1] == pytest.approx(0.75)
    assert cp[2] == pytest.approx(0.0)
    assert cp[3] == pytest.approx(-1.25)


def test_pressure_coefficient_incompressible_invalid():
    with pytest.raises(ValueError):
        pressure_coefficient_incompressible(-10.0, 100.0)

    with pytest.raises(ValueError):
        pressure_coefficient_incompressible(50.0, 0.0)

    with pytest.raises(ValueError):
        pressure_coefficient_incompressible(50.0, -100.0)

    with pytest.raises(TypeError):
        pressure_coefficient_incompressible("bad", 100.0)


# ==============================================================================
# Aerodynamic Forces & Coefficients
# ==============================================================================

def test_lift_force_and_coefficient_inversion():
    cl = 0.55
    q = 2500.0
    s = 16.0

    l = lift_force(cl, q, s)
    assert l == pytest.approx(cl * q * s)

    cl_calc = lift_coefficient(l, q, s)
    assert cl_calc == pytest.approx(cl)


def test_lift_force_invalid():
    with pytest.raises(ValueError):
        lift_force(0.5, -100, 10)

    with pytest.raises(ValueError):
        lift_force(0.5, 1000, 0)

    with pytest.raises(TypeError):
        lift_force(True, 1000, 10)


def test_drag_force_and_coefficient_inversion():
    cd = 0.028
    q = 3000.0
    s = 20.0

    d = drag_force(cd, q, s)
    assert d == pytest.approx(cd * q * s)

    cd_calc = drag_coefficient(d, q, s)
    assert cd_calc == pytest.approx(cd)


def test_drag_force_invalid():
    with pytest.raises(ValueError):
        drag_force(-0.01, 1000, 10)

    with pytest.raises(ValueError):
        drag_coefficient(-10, 1000, 10)


def test_pitching_moment_and_coefficient_inversion():
    cm = -0.08
    q = 1500.0
    s = 12.0
    c = 1.8

    m = pitching_moment(cm, q, s, c)
    assert m == pytest.approx(cm * q * s * c)

    cm_calc = pitching_moment_coefficient(m, q, s, c)
    assert cm_calc == pytest.approx(cm)


def test_pitching_moment_invalid():
    with pytest.raises(ValueError):
        pitching_moment(0.1, 1000, 10, -1)

    with pytest.raises(TypeError):
        pitching_moment(0.1, 1000, "bad", 1)


def test_lift_to_drag_ratio():
    cl = 1.05
    cd = 0.035
    assert lift_to_drag_ratio(cl, cd) == pytest.approx(30.0)

    with pytest.raises(ValueError):
        lift_to_drag_ratio(1.0, 0.0)

    with pytest.raises(ValueError):
        lift_to_drag_ratio(1.0, -0.02)


# ==============================================================================
# Axis Transformations (Body vs Wind)
# ==============================================================================

def test_transformations_zero_alpha():
    cl = 0.6
    cd = 0.03

    cn, ca = wind_to_body_coefficients(cl, cd, alpha=0.0)
    assert cn == pytest.approx(cl)
    assert ca == pytest.approx(cd)

    cl_back, cd_back = body_to_wind_coefficients(cn, ca, alpha=0.0)
    assert cl_back == pytest.approx(cl)
    assert cd_back == pytest.approx(cd)


def test_transformations_roundtrip_radians():
    cl_orig = 0.75
    cd_orig = 0.045
    alpha = 0.12  # rad (~6.87 deg)

    cn, ca = wind_to_body_coefficients(cl_orig, cd_orig, alpha)
    cl_calc, cd_calc = body_to_wind_coefficients(cn, ca, alpha)

    assert cl_calc == pytest.approx(cl_orig)
    assert cd_calc == pytest.approx(cd_orig)


def test_transformations_roundtrip_degrees():
    cl_orig = 1.1
    cd_orig = 0.08
    alpha_deg = 8.5

    cn, ca = wind_to_body_coefficients(cl_orig, cd_orig, alpha_deg, deg=True)
    cl_calc, cd_calc = body_to_wind_coefficients(cn, ca, alpha_deg, deg=True)

    assert cl_calc == pytest.approx(cl_orig)
    assert cd_calc == pytest.approx(cd_orig)


def test_transformations_invalid():
    with pytest.raises(TypeError):
        wind_to_body_coefficients(0.5, 0.02, "bad")

    with pytest.raises(TypeError):
        wind_to_body_coefficients(0.5, 0.02, 5.0, deg="not_bool")


# ==============================================================================
# Skin Friction and Boundary Layer
# ==============================================================================

def test_skin_friction_laminar():
    re = 1e5
    cf = skin_friction_coefficient_laminar(re)
    expected = 1.328 / math.sqrt(1e5)
    assert cf == pytest.approx(expected)


def test_skin_friction_turbulent():
    re = 1e6
    cf_prandtl = skin_friction_coefficient_turbulent(re, model="prandtl")
    assert cf_prandtl == pytest.approx(0.074 / (1e6 ** 0.2))

    cf_schlichting = skin_friction_coefficient_turbulent(re, model="schlichting")
    assert cf_schlichting == pytest.approx(0.455 / (math.log10(1e6) ** 2.58))


def test_skin_friction_coefficient_auto():
    # Subcrítico (laminar)
    cf_lam = skin_friction_coefficient(1e5, regime="auto", re_critical=5e5)
    assert cf_lam == pytest.approx(skin_friction_coefficient_laminar(1e5))

    # Supercrítico (transición)
    cf_trans = skin_friction_coefficient(5e6, regime="auto", re_critical=5e5)
    assert cf_trans > 0.0


def test_skin_friction_invalid():
    with pytest.raises(ValueError):
        skin_friction_coefficient_laminar(-100)

    with pytest.raises(ValueError):
        skin_friction_coefficient_turbulent(1e6, model="unknown")

    with pytest.raises(ValueError):
        skin_friction_coefficient(1e6, regime="invalid_regime")


def test_boundary_layer_thickness():
    x = 1.5
    re_x = 4e4

    # Laminar: 5.0 * x / sqrt(Re_x)
    delta_lam = boundary_layer_thickness(x, re_x, regime="laminar")
    assert delta_lam == pytest.approx(5.0 * 1.5 / math.sqrt(4e4))

    # Turbulento: 0.37 * x / Re_x^0.2
    delta_turb = boundary_layer_thickness(x, re_x, regime="turbulent")
    assert delta_turb == pytest.approx(0.37 * 1.5 / (4e4 ** 0.2))

    # x = 0
    assert boundary_layer_thickness(0.0, 1e5) == pytest.approx(0.0)


def test_boundary_layer_thickness_invalid():
    with pytest.raises(ValueError):
        boundary_layer_thickness(-1.0, 1e5)

    with pytest.raises(ValueError):
        boundary_layer_thickness(1.0, -100)

    with pytest.raises(ValueError):
        boundary_layer_thickness(1.0, 1e5, regime="bad")
