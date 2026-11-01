import math
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



# ============================================================
# monotone_surface
# ============================================================

from pyadinic.modelos.naca_airfoils import monotone_surface
from pyadinic.numerical_methods.interpolation import cubic_spline_interpolation


class TestMonotoneSurface:
    def test_naca_23015_upper_surface(self):
        af = naca5("23015", n_points=100)
        # Verifica que la superficie original tiene no monotonicidad en LE
        assert np.min(np.diff(af.x_upper)) < 0.0

        # Aplica monotone_surface
        xm, ym = monotone_surface(af.x_upper, af.y_upper)

        # Debe ser estrictamente creciente
        assert len(xm) > 2
        assert len(xm) == len(ym)
        assert np.all(np.diff(xm) > 0.0)

        # Permite realizar una interpolacion spline sin errores
        y_interp = cubic_spline_interpolation(0.5, xm, ym)
        assert isinstance(y_interp, float)

    def test_naca_4412_upper_surface(self):
        af = naca4("4412", n_points=100)
        # Verifica que la superficie original no es estrictamente creciente
        assert np.min(np.diff(af.x_upper)) < 0.0

        # Aplica monotone_surface
        xm, ym = monotone_surface(af.x_upper, af.y_upper)

        # Debe ser estrictamente creciente
        assert len(xm) > 2
        assert len(xm) == len(ym)
        assert np.all(np.diff(xm) > 0.0)

        # Permite realizar una interpolacion spline sin errores
        y_interp = cubic_spline_interpolation(0.3, xm, ym)
        assert isinstance(y_interp, float)

    def test_discards_duplicates_within_tolerance(self):
        x = [0.0, 0.1, 0.1 + 1e-10, 0.2, 0.3]
        y = [0.0, 1.0, 1.5, 2.0, 3.0]
        xm, ym = monotone_surface(x, y, tol=1e-9)
        assert len(xm) == 4
        assert np.all(np.diff(xm) > 0.0)

    def test_sorting_unsorted_arrays(self):
        x = [0.5, 0.1, 0.9, 0.3]
        y = [5.0, 1.0, 9.0, 3.0]
        xm, ym = monotone_surface(x, y)
        assert np.all(xm == [0.1, 0.3, 0.5, 0.9])
        assert np.all(ym == [1.0, 3.0, 5.0, 9.0])

    def test_invalid_inputs(self):
        with pytest.raises(TypeError):
            monotone_surface([0.1, True], [0.0, 1.0])
        with pytest.raises(TypeError):
            monotone_surface(["a", "b"], [0.0, 1.0])
        with pytest.raises(ValueError):
            monotone_surface([0.1], [0.1])
        with pytest.raises(ValueError):
            monotone_surface([0.1, 0.2], [0.1])



# ============================================================
# Pruebas B2: Perfiles NACA modificados, thickness_func,
# airfoil_from_coordinates y airfoil_from_selig
# ============================================================

from pyadinic.modelos.naca_airfoils import (
    thickness_distribution_modified,
    airfoil_from_coordinates,
    airfoil_from_selig,
)
from pyadinic.modelos.panel_methods import panel_analysis


class TestNacaModifiedParsers:
    def test_parse_naca4_modified_valid(self):
        m, p, t, le_idx, xmaxt = parse_naca4("0012-64")
        assert m == pytest.approx(0.0)
        assert p == pytest.approx(0.0)
        assert t == pytest.approx(0.12)
        assert le_idx == pytest.approx(6.0)
        assert xmaxt == pytest.approx(0.4)

        m, p, t, le_idx, xmaxt = parse_naca4("2412-34")
        assert m == pytest.approx(0.02)
        assert p == pytest.approx(0.4)
        assert t == pytest.approx(0.12)
        assert le_idx == pytest.approx(3.0)
        assert xmaxt == pytest.approx(0.4)

    def test_parse_naca4_modified_errors(self):
        with pytest.raises(TypeError):
            parse_naca4("0012-6")
        with pytest.raises(TypeError):
            parse_naca4("0012-644")
        with pytest.raises(TypeError):
            parse_naca4("0012-A4")
        with pytest.raises(TypeError):
            parse_naca4("001-64")

    def test_parse_naca5_modified_valid(self):
        res = parse_naca5("23015-64")
        assert len(res) == 9
        cl_des, p_max, third, t, r, k1, k2_ratio, le_idx, xmaxt = res
        assert cl_des == pytest.approx(0.3)
        assert p_max == pytest.approx(0.15)
        assert third == 0
        assert t == pytest.approx(0.15)
        assert le_idx == pytest.approx(6.0)
        assert xmaxt == pytest.approx(0.4)

    def test_parse_naca5_modified_errors(self):
        with pytest.raises(TypeError):
            parse_naca5("23015-6")
        with pytest.raises(TypeError):
            parse_naca5("23015-644")
        with pytest.raises(TypeError):
            parse_naca5("23015-A4")


class TestThicknessDistributionModified:
    def test_values_at_le_and_max_thickness(self):
        t = 0.12
        xmaxt = 0.4
        # En el borde de ataque x=0, semiespesor es 0
        yt_le = thickness_distribution_modified(0.0, t, le_index=6.0, xmaxt=xmaxt)
        assert yt_le == pytest.approx(0.0)

        # En la estacion de espesor maximo x=xmaxt, el semiespesor es t / 2
        yt_max = thickness_distribution_modified(xmaxt, t, le_index=6.0, xmaxt=xmaxt)
        assert yt_max == pytest.approx(t / 2.0, rel=1e-5)

    def test_smoothness_across_boundary(self):
        t = 0.12
        xmaxt = 0.4
        eps = 1e-6
        yt_before = thickness_distribution_modified(xmaxt - eps, t, le_index=6.0, xmaxt=xmaxt)
        yt_after = thickness_distribution_modified(xmaxt + eps, t, le_index=6.0, xmaxt=xmaxt)
        assert yt_before == pytest.approx(yt_after, rel=1e-4)

    def test_le_index_sharp_vs_blunt(self):
        t = 0.12
        x = 0.05
        yt_sharp = thickness_distribution_modified(x, t, le_index=0.0, xmaxt=0.4)
        yt_standard = thickness_distribution_modified(x, t, le_index=6.0, xmaxt=0.4)
        yt_blunt = thickness_distribution_modified(x, t, le_index=9.0, xmaxt=0.4)
        # Mayor radio LE produce mayor espesor cerca del borde de ataque
        assert yt_sharp < yt_standard < yt_blunt

    def test_vectorized_evaluation(self):
        x = np.linspace(0, 1, 50)
        yt = thickness_distribution_modified(x, 0.15, le_index=6.0, xmaxt=0.3)
        assert isinstance(yt, np.ndarray)
        assert len(yt) == 50
        assert np.all(yt >= 0.0)

    def test_errors(self):
        with pytest.raises(ValueError):
            thickness_distribution_modified(-0.01, 0.12)
        with pytest.raises(ValueError):
            thickness_distribution_modified(1.01, 0.12)
        with pytest.raises(ValueError):
            thickness_distribution_modified(0.5, -0.1)
        with pytest.raises(ValueError):
            thickness_distribution_modified(0.5, 0.12, le_index=-1.0)
        with pytest.raises(ValueError):
            thickness_distribution_modified(0.5, 0.12, xmaxt=0.01)
        with pytest.raises(TypeError):
            thickness_distribution_modified(True, 0.12)
        with pytest.raises(TypeError):
            thickness_distribution_modified(0.5, "0.12")


class TestNacaModifiedGeneration:
    def test_naca4_modified_geometry(self):
        foil = naca4("0012-64", n_points=100)
        assert foil.name == "NACA 0012-64"
        assert len(foil.x_upper) == 100
        assert len(foil.y_upper) == 100
        # Simetrico: y_upper == -y_lower
        assert np.allclose(foil.y_upper, -foil.y_lower, atol=1e-12)
        # Espesor maximo cercano a x = 0.4
        idx_max = np.argmax(foil.y_upper)
        assert foil.x_upper[idx_max] == pytest.approx(0.4, abs=0.03)
        assert foil.y_upper[idx_max] == pytest.approx(0.06, abs=1e-3)

    def test_naca5_modified_geometry(self):
        foil = naca5("23015-64", n_points=100)
        assert foil.name == "NACA 23015-64"
        assert len(foil.x_upper) == 100
        assert len(foil.y_upper) == 100
        # Camber positivo
        assert np.max(foil.y_upper) > 0.075

    def test_thickness_func_injection(self):
        # Distribucion eliptica custom de espesor
        custom_func = lambda x, t: (t / 2.0) * math.sqrt(max(0.0, 1.0 - ((x - 0.5) / 0.5) ** 2))
        foil = naca4("0012", n_points=101, thickness_func=custom_func)
        assert foil.name == "NACA 0012"
        # En el centro x=0.5, y_upper debe ser t/2 = 0.06
        idx_mid = np.argmin(np.abs(foil.x_upper - 0.5))
        assert foil.y_upper[idx_mid] == pytest.approx(0.06, abs=1e-3)

    def test_panel_analysis_modified_naca(self):
        res = panel_analysis("0012-64", alpha=0.0)
        assert res.cl == pytest.approx(0.0, abs=1e-3)
        assert res.cd == pytest.approx(0.0, abs=1e-3)


class TestAirfoilFromCoordinates:
    def test_from_separate_surfaces(self):
        x = np.linspace(0, 1, 30)
        y_u = 0.06 * np.sin(np.pi * x)
        y_l = -0.06 * np.sin(np.pi * x)
        foil = airfoil_from_coordinates("CustomFoil", x, y_u, x, y_l)
        assert foil.name == "CustomFoil"
        assert len(foil.x_upper) == 30
        assert np.all(foil.x_upper == x)
        assert np.all(foil.y_upper == y_u)

    def test_from_closed_contour_selig_order(self):
        # Generar un contorno cerrado tipo Selig: TE superior -> LE -> TE inferior
        af_orig = naca4("0012", n_points=50)
        x_contour = np.concatenate([af_orig.x_upper[::-1], af_orig.x_lower[1:]])
        y_contour = np.concatenate([af_orig.y_upper[::-1], af_orig.y_lower[1:]])

        foil = airfoil_from_coordinates("Reconstructed0012", x_contour, y_contour)
        assert foil.name == "Reconstructed0012"
        # Ambas superficies deben ir de LE (x=0) a TE (x=1)
        assert foil.x_upper[0] == pytest.approx(0.0, abs=1e-3)
        assert foil.x_upper[-1] == pytest.approx(1.0, abs=1e-3)
        assert foil.x_lower[0] == pytest.approx(0.0, abs=1e-3)
        assert foil.x_lower[-1] == pytest.approx(1.0, abs=1e-3)
        assert np.mean(foil.y_upper) > np.mean(foil.y_lower)

    def test_errors(self):
        with pytest.raises(TypeError):
            airfoil_from_coordinates(123, [0, 1], [0, 0])
        with pytest.raises(TypeError):
            airfoil_from_coordinates("Foil", [0, True], [0, 0])
        with pytest.raises(ValueError):
            airfoil_from_coordinates("Foil", [0, 1], [0, 0])  # menos de 4 puntos
        with pytest.raises(ValueError):
            airfoil_from_coordinates("Foil", [0, 1], [0, 0], x_lower=[0, 1])  # falta y_lower


class TestAirfoilFromSelig:
    def test_load_from_multiline_string(self):
        selig_data = """E387 Test
1.00000   0.00000
0.90000   0.02000
0.50000   0.07000
0.10000   0.05000
0.00000   0.00000
0.10000  -0.03000
0.50000  -0.03000
0.90000  -0.01000
1.00000   0.00000
"""
        foil = airfoil_from_selig(selig_data, normalize=True)
        assert foil.name == "E387 Test"
        assert foil.x_upper[0] == pytest.approx(0.0)
        assert foil.x_upper[-1] == pytest.approx(1.0)
        assert foil.x_lower[0] == pytest.approx(0.0)
        assert foil.x_lower[-1] == pytest.approx(1.0)
        # Analisis de paneles corre sin problemas sobre el perfil cargado
        state = panel_analysis(foil, alpha=2.0, deg=True)
        assert isinstance(state.cl, float)

    def test_load_from_io(self):
        import io
        buf = io.StringIO("SymmetricAirfoil\n1.0 0.0\n0.5 0.05\n0.0 0.0\n0.5 -0.05\n1.0 0.0\n")
        foil = airfoil_from_selig(buf)
        assert foil.name == "SymmetricAirfoil"
        assert len(foil.x_upper) >= 2
        assert len(foil.x_lower) >= 2

    def test_normalization_scaling(self):
        # Perfil dado en milimetros (cuerda 100 mm)
        data_mm = """ScaledFoil
100.0   0.0
50.0    5.0
0.0     0.0
50.0   -5.0
100.0   0.0
"""
        foil = airfoil_from_selig(data_mm, normalize=True)
        assert foil.x_upper[-1] == pytest.approx(1.0)
        assert np.max(foil.y_upper) == pytest.approx(0.05)  # 5 / 100

    def test_errors(self):
        with pytest.raises(TypeError):
            airfoil_from_selig(12345)
        with pytest.raises(ValueError):
            airfoil_from_selig("Empty\n")
