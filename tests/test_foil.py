import numpy as np
import pytest

from pyadinic.modelos.naca_airfoils import (
    parse_naca4,
    thickness_distribution,
    camber_line,
    camber_line_slope,
    surface_coordinates,
    spacing,
    naca4,
)
from pyadinic.modelos.naca_airfoils import (
    parse_naca5,
    camber_line_5,
    camber_line_slope_5,
    naca5,
)


# ============================================================
# NACA 4-digit parser
# ============================================================

def test_parse_naca4():
    m, p, t = parse_naca4("2412")

    assert m == pytest.approx(0.02)
    assert p == pytest.approx(0.4)
    assert t == pytest.approx(0.12)


def test_parse_naca4_invalid_length():
    with pytest.raises(TypeError):
        parse_naca4("241")


def test_parse_naca4_invalid_characters():
    with pytest.raises(TypeError):
        parse_naca4("24A2")


def test_parse_naca4_invalid_camber_position():
    with pytest.raises(ValueError):
        parse_naca4("2012")


# ============================================================
# Thickness distribution
# ============================================================

def test_thickness_distribution():
    y_t = thickness_distribution(0.5, 0.12)

    assert y_t == pytest.approx(0.05294, rel=1e-3)


def test_thickness_distribution_bounds():
    with pytest.raises(ValueError):
        thickness_distribution(-0.1, 0.12)

    with pytest.raises(ValueError):
        thickness_distribution(1.1, 0.12)


# ============================================================
# Camber line
# ============================================================

def test_camber_line_at_leading_edge():
    y_c = camber_line(0.0, 0.02, 0.4)

    assert y_c == pytest.approx(0.0)


def test_camber_line_at_max_camber():
    y_c = camber_line(0.4, 0.02, 0.4)

    assert y_c == pytest.approx(0.02)


def test_camber_line_at_trailing_edge():
    y_c = camber_line(1.0, 0.02, 0.4)

    assert y_c == pytest.approx(0.0)


# ============================================================
# Camber line slope
# ============================================================

def test_camber_line_slope_at_max_camber():
    slope = camber_line_slope(0.4, 0.02, 0.4)

    assert slope == pytest.approx(0.0)


def test_camber_line_slope_leading_edge():
    slope = camber_line_slope(0.0, 0.02, 0.4)

    assert slope == pytest.approx(0.1)


def test_camber_line_slope_trailing_edge():
    slope = camber_line_slope(1.0, 0.02, 0.4)

    assert slope == pytest.approx(-0.0666667)


# ============================================================
# Surface coordinates
# ============================================================

def test_surface_coordinates():
    x_upper, y_upper, x_lower, y_lower = surface_coordinates(
        0.5, 0.02, 0.4, 0.12
    )

    assert x_upper == pytest.approx(0.5005881887)
    assert y_upper > 0

    assert x_lower == pytest.approx(0.4994118113)
    assert y_lower < 0


# ============================================================
# Spacing
# ============================================================

def test_spacing_regular():
    x = spacing(5, "regular")

    assert isinstance(x, np.ndarray)
    assert len(x) == 5

    assert np.allclose(
        x,
        [0.0, 0.25, 0.5, 0.75, 1.0]
    )


def test_spacing_cosine():
    x = spacing(5, "cosine")

    assert isinstance(x, np.ndarray)
    assert len(x) == 5

    assert x[0] == pytest.approx(0.0)
    assert x[-1] == pytest.approx(1.0)

    assert x[2] == pytest.approx(0.5)


def test_spacing_cosine_bounds():
    x = spacing(100, "cosine")

    assert np.all(x >= 0)
    assert np.all(x <= 1)


# ============================================================
# NACA 4-digit airfoil
# ============================================================

def test_naca4_returns_airfoil():
    airfoil = naca4("2412")

    assert airfoil.name == "NACA 2412"


def test_naca4_number_of_points():
    n_points = 50

    airfoil = naca4(
        "2412",
        n_points=n_points
    )

    assert len(airfoil.x_upper) == n_points
    assert len(airfoil.y_upper) == n_points
    assert len(airfoil.x_lower) == n_points
    assert len(airfoil.y_lower) == n_points


def test_naca4_arrays():
    airfoil = naca4(
        "2412",
        n_points=50
    )

    assert isinstance(airfoil.x_upper, np.ndarray)
    assert isinstance(airfoil.y_upper, np.ndarray)
    assert isinstance(airfoil.x_lower, np.ndarray)
    assert isinstance(airfoil.y_lower, np.ndarray)


def test_naca4_chord_scaling():
    airfoil_1 = naca4(
        "2412",
        chord=1.0,
        n_points=50
    )

    airfoil_2 = naca4(
        "2412",
        chord=2.0,
        n_points=50
    )

    assert np.allclose(
        airfoil_2.x_upper,
        2 * airfoil_1.x_upper
    )

    assert np.allclose(
        airfoil_2.y_upper,
        2 * airfoil_1.y_upper
    )

    assert np.allclose(
        airfoil_2.x_lower,
        2 * airfoil_1.x_lower
    )

    assert np.allclose(
        airfoil_2.y_lower,
        2 * airfoil_1.y_lower
    )


def test_naca4_leading_edge():
    airfoil = naca4(
        "2412",
        n_points=50
    )

    assert airfoil.x_upper[0] == pytest.approx(0.0)
    assert airfoil.y_upper[0] == pytest.approx(0.0)

    assert airfoil.x_lower[0] == pytest.approx(0.0)
    assert airfoil.y_lower[0] == pytest.approx(0.0)


def test_naca4_upper_surface_above_lower():
    airfoil = naca4(
        "2412",
        n_points=50
    )

    assert np.all(
        airfoil.y_upper >= airfoil.y_lower
    )

# ============================================================
# NACA 5-digit parser
# ============================================================

def test_parse_naca5_standard():
    (
        design_lift_coefficient,
        max_camber_position,
        camber_type,
        thickness,
        r,
        k1,
        k2_ratio,
    ) = parse_naca5("23012")

    assert design_lift_coefficient == pytest.approx(0.30)
    assert max_camber_position == pytest.approx(0.15)
    assert camber_type == 0
    assert thickness == pytest.approx(0.12)

    assert r == pytest.approx(0.2025)
    assert k1 == pytest.approx(15.957)
    assert k2_ratio == pytest.approx(0.0)


def test_parse_naca5_reflex():
    (
        design_lift_coefficient,
        max_camber_position,
        camber_type,
        thickness,
        r,
        k1,
        k2_ratio,
    ) = parse_naca5("23112")

    assert design_lift_coefficient == pytest.approx(0.30)
    assert max_camber_position == pytest.approx(0.15)
    assert camber_type == 1
    assert thickness == pytest.approx(0.12)

    assert r == pytest.approx(0.217)
    assert k1 == pytest.approx(15.793)
    assert k2_ratio == pytest.approx(0.00677)


def test_parse_naca5_invalid_length():
    with pytest.raises(TypeError):
        parse_naca5("2301")


def test_parse_naca5_invalid_characters():
    with pytest.raises(TypeError):
        parse_naca5("23A12")


def test_parse_naca5_invalid_first_digit():
    with pytest.raises(ValueError):
        parse_naca5("03012")


def test_parse_naca5_invalid_second_digit():
    with pytest.raises(ValueError):
        parse_naca5("20012")


def test_parse_naca5_invalid_camber_type():
    with pytest.raises(ValueError):
        parse_naca5("23212")


def test_parse_naca5_invalid_reflex_combination():
    with pytest.raises(ValueError):
        parse_naca5("21112")


# ============================================================
# NACA 5-digit camber line
# ============================================================

def test_camber_line_5_leading_edge():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        _
    ) = parse_naca5("23012")

    y_c = camber_line_5(
        0.0,
        r,
        k1,
    )

    assert y_c == pytest.approx(0.0)


def test_camber_line_5_trailing_edge():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        _
    ) = parse_naca5("23012")

    y_c = camber_line_5(
        1.0,
        r,
        k1,
    )

    assert y_c == pytest.approx(0.0)


def test_camber_line_5_continuity():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        _
    ) = parse_naca5("23012")

    y_before = camber_line_5(
        r - 1e-8,
        r,
        k1,
    )

    y_after = camber_line_5(
        r + 1e-8,
        r,
        k1,
    )

    assert y_before == pytest.approx(y_after, abs=1e-7)


# ============================================================
# NACA 5-digit camber slope
# ============================================================

def test_camber_line_slope_5_continuity():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        _
    ) = parse_naca5("23012")

    slope_before = camber_line_slope_5(
        r - 1e-8,
        r,
        k1,
    )

    slope_after = camber_line_slope_5(
        r + 1e-8,
        r,
        k1,
    )

    assert slope_before == pytest.approx(
        slope_after,
        abs=1e-7
    )


def test_camber_line_slope_5_at_trailing_edge():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        _
    ) = parse_naca5("23012")

    slope = camber_line_slope_5(
        1.0,
        r,
        k1,
    )

    assert slope == pytest.approx(
        -k1 * r**3 / 6
    )


# ============================================================
# NACA 5-digit reflex camber
# ============================================================

def test_camber_line_5_reflex_leading_edge():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        k2_ratio,
    ) = parse_naca5("23112")

    y_c = camber_line_5(
        0.0,
        r,
        k1,
        reflex=True,
        k2_ratio=k2_ratio,
    )

    assert y_c == pytest.approx(0.0)


def test_camber_line_5_reflex_trailing_edge():
    (
        _,
        _,
        _,
        _,
        r,
        k1,
        k2_ratio,
    ) = parse_naca5("23112")

    y_c = camber_line_5(
        1.0,
        r,
        k1,
        reflex=True,
        k2_ratio=k2_ratio,
    )

    assert y_c == pytest.approx(0.0)


# ============================================================
# NACA 5-digit airfoil
# ============================================================

def test_naca5_returns_airfoil():
    airfoil = naca5("23012")

    assert airfoil.name == "NACA 23012"


def test_naca5_number_of_points():
    n_points = 50

    airfoil = naca5(
        "23012",
        n_points=n_points,
    )

    assert len(airfoil.x_upper) == n_points
    assert len(airfoil.y_upper) == n_points
    assert len(airfoil.x_lower) == n_points
    assert len(airfoil.y_lower) == n_points


def test_naca5_arrays():
    airfoil = naca5(
        "23012",
        n_points=50,
    )

    assert isinstance(airfoil.x_upper, np.ndarray)
    assert isinstance(airfoil.y_upper, np.ndarray)
    assert isinstance(airfoil.x_lower, np.ndarray)
    assert isinstance(airfoil.y_lower, np.ndarray)


def test_naca5_chord_scaling():
    airfoil_1 = naca5(
        "23012",
        chord=1.0,
        n_points=50,
    )

    airfoil_2 = naca5(
        "23012",
        chord=2.0,
        n_points=50,
    )

    assert np.allclose(
        airfoil_2.x_upper,
        2 * airfoil_1.x_upper,
    )

    assert np.allclose(
        airfoil_2.y_upper,
        2 * airfoil_1.y_upper,
    )

    assert np.allclose(
        airfoil_2.x_lower,
        2 * airfoil_1.x_lower,
    )

    assert np.allclose(
        airfoil_2.y_lower,
        2 * airfoil_1.y_lower,
    )


def test_naca5_leading_edge():
    airfoil = naca5(
        "23012",
        n_points=50,
    )

    assert airfoil.x_upper[0] == pytest.approx(0.0)
    assert airfoil.y_upper[0] == pytest.approx(0.0)

    assert airfoil.x_lower[0] == pytest.approx(0.0)
    assert airfoil.y_lower[0] == pytest.approx(0.0)


def test_naca5_upper_surface_above_lower():
    airfoil = naca5(
        "23012",
        n_points=50,
    )

    assert np.all(
        airfoil.y_upper >= airfoil.y_lower
    )


def test_naca5_reflex():
    airfoil = naca5(
        "23112",
        n_points=50,
    )

    assert airfoil.name == "NACA 23112"

    assert isinstance(
        airfoil.x_upper,
        np.ndarray,
    )

    assert isinstance(
        airfoil.x_lower,
        np.ndarray,
    )


# ============================================================
# Symmetric NACA 4-digit airfoil (e.g. NACA 0012)
# ============================================================

def test_parse_naca4_symmetric():
    m, p, t = parse_naca4("0012")

    assert m == pytest.approx(0.0)
    assert p == pytest.approx(0.0)
    assert t == pytest.approx(0.12)


def test_naca4_symmetric():
    airfoil = naca4("0012", n_points=50)

    assert airfoil.name == "NACA 0012"
    # Upper and lower camber should be symmetric
    assert np.allclose(airfoil.y_upper, -airfoil.y_lower)
    assert np.allclose(airfoil.x_upper, airfoil.x_lower)


def test_naca4_regular_spacing():
    airfoil = naca4("2412", n_points=25, spacing_type="regular")

    assert len(airfoil.x_upper) == 25
    assert airfoil.name == "NACA 2412"


def test_naca5_regular_spacing():
    airfoil = naca5("23012", n_points=25, spacing_type="regular")

    assert len(airfoil.x_upper) == 25
    assert airfoil.name == "NACA 23012"