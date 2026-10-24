import math
import pytest
import numpy as np

from pyadinic.modelos.thin_airfoil import (
    fourier_coefficients,
    thin_airfoil_alpha_0,
    thin_airfoil_cl,
    thin_airfoil_cm_c4,
    thin_airfoil_cm_le,
    thin_airfoil_center_of_pressure,
    thin_airfoil_delta_cp,
    thin_airfoil_theory,
    thin_airfoil_naca4,
    thin_airfoil_naca5,
)
from pyadinic.core.dataclass import ThinAirfoilState


# ==============================================================================
# Helpers
# ==============================================================================

# Perfil simétrico NACA 0012: curva de curvatura nula → todas las magnitudes
# son exactamente calculables analíticamente.
NACA0012 = "0012"

# NACA 2412: caso estándar con curvatura. Valores de referencia tomados de la
# literatura clásica (Abbott & Von Doenhoff, Theory of Wing Sections):
#   alpha_0 ≈ -2.08°   (≈ -0.0363 rad)
#   cm_c/4  ≈ -0.053
NACA2412 = "2412"
NACA23012 = "23012"


# ==============================================================================
# fourier_coefficients
# ==============================================================================

class TestFourierCoefficients:

    def test_symmetric_all_coefficients_zero(self):
        # Perfil simétrico: pendiente de curvatura nula ⟹ A0=0, An=0
        a0, an_list = fourier_coefficients(naca=NACA0012, n_coefficients=3)
        assert a0 == pytest.approx(0.0, abs=1e-6)
        for an in an_list:
            assert an == pytest.approx(0.0, abs=1e-6)

    def test_naca2412_a1_positive(self):
        # Para una curvatura positiva A1 debe ser positivo
        _, an_list = fourier_coefficients(naca=NACA2412, n_coefficients=2)
        assert an_list[0] > 0.0

    def test_custom_slope_function(self):
        # Pendiente constante C: A0 = C, A_n = 0 para n >= 1
        C = 0.15
        a0, an_list = fourier_coefficients(slope_func=lambda x: C, n_coefficients=2)
        assert a0 == pytest.approx(C, rel=1e-4)
        for an in an_list:
            assert an == pytest.approx(0.0, abs=1e-4)

    def test_both_slope_func_and_naca_raises(self):
        with pytest.raises(ValueError):
            fourier_coefficients(slope_func=lambda x: 0.1, naca=NACA2412)

    def test_neither_raises(self):
        with pytest.raises(ValueError):
            fourier_coefficients()

    def test_invalid_naca_type(self):
        with pytest.raises(TypeError):
            fourier_coefficients(naca=2412)

    def test_invalid_naca_length(self):
        with pytest.raises(ValueError):
            fourier_coefficients(naca="123")

    def test_invalid_n_coefficients_type(self):
        with pytest.raises(TypeError):
            fourier_coefficients(naca=NACA2412, n_coefficients=2.5)

    def test_invalid_n_coefficients_value(self):
        with pytest.raises(ValueError):
            fourier_coefficients(naca=NACA2412, n_coefficients=0)

    def test_invalid_n_points_type(self):
        with pytest.raises(TypeError):
            fourier_coefficients(naca=NACA2412, n_points=50.0)

    def test_invalid_n_points_value(self):
        with pytest.raises(ValueError):
            fourier_coefficients(naca=NACA2412, n_points=3)

    def test_slope_func_not_callable_raises(self):
        with pytest.raises(TypeError):
            fourier_coefficients(slope_func="not_callable")


# ==============================================================================
# thin_airfoil_alpha_0
# ==============================================================================

class TestThinAirfoilAlpha0:

    def test_symmetric_alpha_0_is_zero(self):
        a0 = thin_airfoil_alpha_0(naca=NACA0012)
        assert a0 == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_alpha_0_deg_is_zero(self):
        a0_deg = thin_airfoil_alpha_0(naca=NACA0012, deg=True)
        assert a0_deg == pytest.approx(0.0, abs=1e-5)

    def test_naca2412_alpha_0_rad(self):
        # Valor de referencia: ≈ -2.08° ≈ -0.0363 rad
        a0 = thin_airfoil_alpha_0(naca=NACA2412)
        assert a0 == pytest.approx(-0.03624, rel=1e-2)

    def test_naca2412_alpha_0_deg(self):
        a0_deg = thin_airfoil_alpha_0(naca=NACA2412, deg=True)
        assert a0_deg == pytest.approx(-2.077, rel=1e-2)

    def test_camber_increases_magnitude_of_alpha_0(self):
        # NACA 4412 tiene mayor curvatura que NACA 2412 → |alpha_0| mayor
        a0_2412 = thin_airfoil_alpha_0(naca="2412")
        a0_4412 = thin_airfoil_alpha_0(naca="4412")
        assert abs(a0_4412) > abs(a0_2412)

    def test_naca5_alpha_0_negative(self):
        a0 = thin_airfoil_alpha_0(naca=NACA23012)
        assert a0 < 0.0

    def test_invalid_deg(self):
        with pytest.raises(TypeError):
            thin_airfoil_alpha_0(naca=NACA2412, deg="yes")


# ==============================================================================
# thin_airfoil_cl
# ==============================================================================

class TestThinAirfoilCl:

    def test_symmetric_at_zero_alpha_is_zero(self):
        cl = thin_airfoil_cl(0.0, naca=NACA0012)
        assert cl == pytest.approx(0.0, abs=1e-6)

    def test_lift_slope_is_two_pi(self):
        # Δcl / Δα = 2π para cualquier perfil
        alpha1 = 0.05
        alpha2 = 0.10
        cl1 = thin_airfoil_cl(alpha1, naca=NACA2412)
        cl2 = thin_airfoil_cl(alpha2, naca=NACA2412)
        slope = (cl2 - cl1) / (alpha2 - alpha1)
        assert slope == pytest.approx(2.0 * math.pi, rel=1e-4)

    def test_symmetric_lift_slope_two_pi(self):
        alpha1 = 0.05
        alpha2 = 0.10
        cl1 = thin_airfoil_cl(alpha1, naca=NACA0012)
        cl2 = thin_airfoil_cl(alpha2, naca=NACA0012)
        slope = (cl2 - cl1) / (alpha2 - alpha1)
        assert slope == pytest.approx(2.0 * math.pi, rel=1e-4)

    def test_naca2412_at_alpha_0_is_zero(self):
        alpha_0_rad = thin_airfoil_alpha_0(naca=NACA2412)
        cl = thin_airfoil_cl(alpha_0_rad, naca=NACA2412)
        assert cl == pytest.approx(0.0, abs=1e-8)

    def test_degrees_radians_equivalence(self):
        cl_rad = thin_airfoil_cl(math.radians(4.0), naca=NACA2412)
        cl_deg = thin_airfoil_cl(4.0, naca=NACA2412, deg=True)
        assert cl_rad == pytest.approx(cl_deg, rel=1e-8)

    def test_positive_alpha_gives_positive_cl(self):
        cl = thin_airfoil_cl(0.1, naca=NACA2412)
        assert cl > 0.0

    def test_custom_slope_func_flat_plate(self):
        # Placa plana: dy_c/dx = 0 → cl = 2π * alpha
        alpha = 0.07
        cl = thin_airfoil_cl(alpha, slope_func=lambda x: 0.0)
        assert cl == pytest.approx(2.0 * math.pi * alpha, rel=1e-4)

    def test_invalid_alpha_type(self):
        with pytest.raises(TypeError):
            thin_airfoil_cl("bad", naca=NACA2412)

    def test_invalid_alpha_bool(self):
        with pytest.raises(TypeError):
            thin_airfoil_cl(True, naca=NACA2412)


# ==============================================================================
# thin_airfoil_cm_c4
# ==============================================================================

class TestThinAirfoilCmC4:

    def test_symmetric_cm_c4_is_zero(self):
        # Perfil simétrico: A1 = A2 = 0 → cm_c/4 = 0
        cm = thin_airfoil_cm_c4(naca=NACA0012)
        assert cm == pytest.approx(0.0, abs=1e-6)

    def test_naca2412_cm_c4_negative(self):
        # Para NACA 2412 con curvatura positiva: cm_c/4 ≈ -0.053 (cabeceo nariz abajo)
        cm = thin_airfoil_cm_c4(naca=NACA2412)
        assert cm < 0.0
        assert cm == pytest.approx(-0.0531, rel=2e-2)

    def test_cm_c4_independent_of_alpha(self):
        # cm_c/4 es constante con alpha (característica del c/4 como centro aerodinámico)
        cm1 = thin_airfoil_cm_c4(naca=NACA2412)
        cm2 = thin_airfoil_cm_c4(naca=NACA2412)
        assert cm1 == pytest.approx(cm2)

    def test_flat_plate_cm_c4_is_zero(self):
        cm = thin_airfoil_cm_c4(slope_func=lambda x: 0.0)
        assert cm == pytest.approx(0.0, abs=1e-8)

    def test_naca5_cm_c4(self):
        cm = thin_airfoil_cm_c4(naca=NACA23012)
        # El perfil NACA 23012 tiene curvatura: cm_c/4 debe ser negativo
        assert cm < 0.0


# ==============================================================================
# thin_airfoil_cm_le
# ==============================================================================

class TestThinAirfoilCmLe:

    def test_cm_le_relation_to_cm_c4_and_cl(self):
        # cm_le = cm_c/4 - 0.25 * cl
        alpha = 0.08
        cl = thin_airfoil_cl(alpha, naca=NACA2412)
        cm_c4 = thin_airfoil_cm_c4(naca=NACA2412)
        cm_le_expected = cm_c4 - 0.25 * cl

        cm_le = thin_airfoil_cm_le(alpha, naca=NACA2412)
        assert cm_le == pytest.approx(cm_le_expected, rel=1e-6)

    def test_symmetric_flat_plate_cm_le(self):
        # Placa plana, alpha = 5° → cm_le = -π * alpha / 2
        alpha = math.radians(5.0)
        cm_le = thin_airfoil_cm_le(alpha, slope_func=lambda x: 0.0)
        expected = -math.pi * alpha / 2.0
        assert cm_le == pytest.approx(expected, rel=1e-4)

    def test_degrees_radians_equivalence(self):
        cm_rad = thin_airfoil_cm_le(math.radians(5.0), naca=NACA2412)
        cm_deg = thin_airfoil_cm_le(5.0, naca=NACA2412, deg=True)
        assert cm_rad == pytest.approx(cm_deg, rel=1e-8)


# ==============================================================================
# thin_airfoil_center_of_pressure
# ==============================================================================

class TestThinAirfoilCenterOfPressure:

    def test_symmetric_cp_at_quarter_chord(self):
        # Perfil simétrico: x_cp = c/4 = 0.25 (independiente de alpha > 0)
        xcp = thin_airfoil_center_of_pressure(0.1, naca=NACA0012)
        assert xcp == pytest.approx(0.25, abs=1e-5)

    def test_flat_plate_cp_at_quarter_chord(self):
        xcp = thin_airfoil_center_of_pressure(0.1, slope_func=lambda x: 0.0)
        assert xcp == pytest.approx(0.25, abs=1e-5)

    def test_cambered_cp_behind_quarter_chord(self):
        # Con curvatura (cm_c/4 < 0), x_cp > 0.25
        xcp = thin_airfoil_center_of_pressure(0.1, naca=NACA2412)
        assert xcp > 0.25

    def test_cp_moves_forward_with_increasing_alpha(self):
        # Para un perfil curvado, x_cp se mueve hacia adelante al aumentar alpha
        xcp_small = thin_airfoil_center_of_pressure(0.02, naca=NACA2412)
        xcp_large = thin_airfoil_center_of_pressure(0.20, naca=NACA2412)
        assert xcp_small > xcp_large

    def test_degrees_radians_equivalence(self):
        xcp_rad = thin_airfoil_center_of_pressure(math.radians(5.0), naca=NACA2412)
        xcp_deg = thin_airfoil_center_of_pressure(5.0, naca=NACA2412, deg=True)
        assert xcp_rad == pytest.approx(xcp_deg, rel=1e-8)


# ==============================================================================
# thin_airfoil_delta_cp
# ==============================================================================

class TestThinAirfoilDeltaCp:

    def test_trailing_edge_delta_cp_is_zero(self):
        delta = thin_airfoil_delta_cp(1.0, alpha=0.1, naca=NACA2412)
        assert delta == pytest.approx(0.0, abs=1e-8)

    def test_leading_edge_delta_cp_is_infinite_for_nonzero_a0(self):
        # Con alpha != alpha_0, A0 != 0, delta_cp → ∞ en x=0
        val = thin_airfoil_delta_cp(0.0, alpha=0.1, naca=NACA2412)
        assert val == float("inf")

    def test_midchord_positive_delta_cp(self):
        delta = thin_airfoil_delta_cp(0.5, alpha=0.1, naca=NACA2412)
        assert delta > 0.0

    def test_array_input(self):
        x_arr = np.array([0.1, 0.5, 0.9])
        delta = thin_airfoil_delta_cp(x_arr, alpha=0.05, naca=NACA2412)
        assert isinstance(delta, np.ndarray)
        assert len(delta) == 3
        assert all(d > 0 for d in delta)

    def test_x_out_of_range_raises(self):
        with pytest.raises(ValueError):
            thin_airfoil_delta_cp(1.5, alpha=0.1, naca=NACA2412)

    def test_invalid_alpha_type(self):
        with pytest.raises(TypeError):
            thin_airfoil_delta_cp(0.5, alpha="bad", naca=NACA2412)


# ==============================================================================
# thin_airfoil_theory  (resultado integrado → ThinAirfoilState)
# ==============================================================================

class TestThinAirfoilTheory:

    def test_returns_thin_airfoil_state(self):
        result = thin_airfoil_theory(0.0, naca=NACA2412)
        assert isinstance(result, ThinAirfoilState)

    def test_lift_slope_is_two_pi(self):
        result = thin_airfoil_theory(0.1, naca=NACA2412)
        assert result.lift_slope == pytest.approx(2.0 * math.pi, rel=1e-8)

    def test_alpha_0_matches_standalone(self):
        result = thin_airfoil_theory(0.05, naca=NACA2412)
        a0_standalone = thin_airfoil_alpha_0(naca=NACA2412)
        assert result.alpha_0 == pytest.approx(a0_standalone, rel=1e-8)

    def test_cl_matches_standalone(self):
        alpha = 0.07
        result = thin_airfoil_theory(alpha, naca=NACA2412)
        cl_standalone = thin_airfoil_cl(alpha, naca=NACA2412)
        assert result.cl == pytest.approx(cl_standalone, rel=1e-8)

    def test_cm_c4_matches_standalone(self):
        result = thin_airfoil_theory(0.05, naca=NACA2412)
        cm_standalone = thin_airfoil_cm_c4(naca=NACA2412)
        assert result.cm_c4 == pytest.approx(cm_standalone, rel=1e-8)

    def test_cm_le_is_cm_c4_minus_quarter_cl(self):
        result = thin_airfoil_theory(0.1, naca=NACA2412)
        assert result.cm_le == pytest.approx(result.cm_c4 - 0.25 * result.cl, rel=1e-8)

    def test_center_of_pressure_matches_standalone(self):
        alpha = 0.08
        result = thin_airfoil_theory(alpha, naca=NACA2412)
        xcp_standalone = thin_airfoil_center_of_pressure(alpha, naca=NACA2412)
        assert result.center_of_pressure == pytest.approx(xcp_standalone, rel=1e-8)

    def test_a0_stored_correctly(self):
        # a0 = alpha - A0_camber
        result = thin_airfoil_theory(0.0, naca=NACA2412)
        a0_camber_ref, an_list = fourier_coefficients(naca=NACA2412, n_coefficients=2)
        expected_a0 = 0.0 - a0_camber_ref
        assert result.a0 == pytest.approx(expected_a0, rel=1e-8)

    def test_naca0012_symmetric_properties(self):
        result = thin_airfoil_theory(0.0, naca=NACA0012)
        assert result.alpha_0 == pytest.approx(0.0, abs=1e-6)
        assert result.cm_c4 == pytest.approx(0.0, abs=1e-6)
        assert result.cl == pytest.approx(0.0, abs=1e-6)

    def test_deg_flag_gives_same_result(self):
        alpha_deg = 5.0
        alpha_rad = math.radians(alpha_deg)
        result_rad = thin_airfoil_theory(alpha_rad, naca=NACA2412)
        result_deg = thin_airfoil_theory(alpha_deg, naca=NACA2412, deg=True)
        assert result_rad.cl == pytest.approx(result_deg.cl, rel=1e-8)
        assert result_rad.cm_c4 == pytest.approx(result_deg.cm_c4, rel=1e-8)

    def test_naca5_theory(self):
        result = thin_airfoil_theory(0.0, naca=NACA23012)
        assert isinstance(result, ThinAirfoilState)
        assert result.cl > 0.0        # NACA 23012 con alfa=0 sustenta
        assert result.cm_c4 < 0.0    # Momento nariz abajo para perfil curvado

    def test_invalid_alpha_type(self):
        with pytest.raises(TypeError):
            thin_airfoil_theory("bad", naca=NACA2412)

    def test_invalid_deg(self):
        with pytest.raises(TypeError):
            thin_airfoil_theory(0.0, naca=NACA2412, deg=0)


# ==============================================================================
# thin_airfoil_naca4 / thin_airfoil_naca5 (atajos)
# ==============================================================================

class TestThinAirfoilNacaAliases:

    def test_naca4_shortcut_matches_theory(self):
        result_alias = thin_airfoil_naca4("2412", alpha=0.1)
        result_theory = thin_airfoil_theory(0.1, naca="2412")
        assert result_alias.cl == pytest.approx(result_theory.cl, rel=1e-8)
        assert result_alias.cm_c4 == pytest.approx(result_theory.cm_c4, rel=1e-8)

    def test_naca5_shortcut_matches_theory(self):
        result_alias = thin_airfoil_naca5("23012", alpha=0.1)
        result_theory = thin_airfoil_theory(0.1, naca="23012")
        assert result_alias.cl == pytest.approx(result_theory.cl, rel=1e-8)
        assert result_alias.cm_c4 == pytest.approx(result_theory.cm_c4, rel=1e-8)

    def test_naca4_returns_state(self):
        assert isinstance(thin_airfoil_naca4("0012"), ThinAirfoilState)

    def test_naca5_returns_state(self):
        assert isinstance(thin_airfoil_naca5("23012"), ThinAirfoilState)

    def test_naca4_deg_flag(self):
        result_deg = thin_airfoil_naca4("2412", alpha=5.0, deg=True)
        result_rad = thin_airfoil_naca4("2412", alpha=math.radians(5.0))
        assert result_deg.cl == pytest.approx(result_rad.cl, rel=1e-8)

    def test_naca5_deg_flag(self):
        result_deg = thin_airfoil_naca5("23012", alpha=3.0, deg=True)
        result_rad = thin_airfoil_naca5("23012", alpha=math.radians(3.0))
        assert result_deg.cl == pytest.approx(result_rad.cl, rel=1e-8)

    def test_naca4_with_naca_prefix(self):
        # El código de 4 dígitos también debe aceptar "NACA 2412"
        result = thin_airfoil_theory(0.0, naca="NACA 2412")
        assert isinstance(result, ThinAirfoilState)
