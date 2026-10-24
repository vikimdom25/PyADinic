import math
import numpy as np
import pytest

from pyadinic.core.dataclass import Airfoil, PanelState
from pyadinic.modelos.naca_airfoils import naca4, naca5
from pyadinic.modelos.thin_airfoil import thin_airfoil_naca4
from pyadinic.modelos.panel_methods import (
    panel_geometry,
    influence_matrix,
    panel_solve,
    panel_pressure,
    panel_forces,
    panel_analysis,
    panel_polar,
    panel_naca4,
    panel_naca5,
    _tangential_and_normal,
)


# ==============================================================================
# panel_geometry
# ==============================================================================

class TestPanelGeometry:
    def test_returns_two_arrays_same_length(self):
        xp, yp = panel_geometry(naca4("0012", n_points=50))
        assert xp.shape == yp.shape
        assert xp.ndim == 1

    def test_node_count(self):
        # (n_points) extradós + (n_points - 1) intradós
        xp, _ = panel_geometry(naca4("0012", n_points=50))
        assert len(xp) == 99

    def test_contour_is_counter_clockwise(self):
        xp, yp = panel_geometry(naca4("2412", n_points=60))
        area = 0.5 * np.sum(xp[:-1] * yp[1:] - xp[1:] * yp[:-1])
        assert area > 0

    def test_starts_at_trailing_edge_and_passes_leading_edge(self):
        xp, _ = panel_geometry(naca4("0012", n_points=50))
        assert xp[0] == pytest.approx(1.0, abs=1e-3)
        assert xp[-1] == pytest.approx(1.0, abs=1e-3)
        assert xp.min() == pytest.approx(0.0, abs=1e-12)

    def test_invalid_type(self):
        with pytest.raises(TypeError):
            panel_geometry("0012")

    def test_too_few_points(self):
        af = Airfoil("x", np.array([0.0]), np.array([0.0]),
                     np.array([0.0]), np.array([0.0]))
        with pytest.raises(ValueError):
            panel_geometry(af)


# ==============================================================================
# influence_matrix / panel_solve
# ==============================================================================

class TestInfluenceAndSolve:
    def setup_method(self):
        self.xp, self.yp = panel_geometry(naca4("0012", n_points=40))
        self.n = len(self.xp) - 1

    def test_shapes(self):
        A, b, xc, yc, nx, ny, tx, ty, length = influence_matrix(self.xp, self.yp, 0.1)
        # N fuentes + 1 vórtice = N+1 incógnitas; N no-penetración + 1 Kutta
        assert A.shape == (self.n + 1, self.n + 1)
        assert b.shape == (self.n + 1,)
        for arr in (xc, yc, nx, ny, tx, ty, length):
            assert arr.shape == (self.n,)

    def test_normals_are_unit_and_point_outward(self):
        _, _, xc, yc, nx, ny, tx, ty, _ = influence_matrix(self.xp, self.yp, 0.0)
        assert np.allclose(nx**2 + ny**2, 1.0)
        assert np.allclose(nx * tx + ny * ty, 0.0)
        cx, cy = 0.4, 0.0
        assert np.all((xc - cx) * nx + (yc - cy) * ny > 0)

    def test_source_self_influence_is_one_half(self):
        A, *_ = influence_matrix(self.xp, self.yp, 0.0)
        assert np.allclose(np.diag(A)[: self.n], 0.5)

    def test_rhs_is_free_stream_normal_component(self):
        a = 0.2
        _, b, _, _, nx, ny, tx, ty, _ = influence_matrix(self.xp, self.yp, a)
        assert np.allclose(b[:-1], -(math.cos(a) * nx + math.sin(a) * ny))
        expected_kutta = -(math.cos(a) * (tx[0] + tx[-1]) + math.sin(a) * (ty[0] + ty[-1]))
        assert b[-1] == pytest.approx(expected_kutta)

    def test_solve_returns_expected_structure(self):
        sigma, gamma, A, b, *_ = panel_solve(self.xp, self.yp, 0.1)
        assert sigma.shape == (self.n,)
        assert isinstance(gamma, float)
        x = np.append(sigma, gamma)
        assert np.allclose(A @ x, b, atol=1e-9)

    def test_source_strengths_sum_to_zero(self):
        # Cuerpo cerrado: flujo neto de las fuentes ~nulo (el borde de salida
        # romo de los NACA deja un pequeño hueco sin panel)
        xp, yp = self.xp, self.yp
        sigma, *_ = panel_solve(xp, yp, 0.1)
        length = np.hypot(np.diff(xp), np.diff(yp))
        assert float(np.sum(sigma * length)) == pytest.approx(0.0, abs=1e-2)

    def test_kutta_condition_equal_trailing_edge_speeds(self):
        xp, yp = self.xp, self.yp
        sigma, gamma, _, _, _, _, _, _, tx, ty, _ = panel_solve(xp, yp, 0.1)
        _, vt = panel_pressure(sigma, gamma, xp, yp, 0.1, tx, ty)
        assert vt[0] + vt[-1] == pytest.approx(0.0, abs=1e-9)

    def test_no_penetration_through_surface(self):
        a = 0.15
        xp, yp = self.xp, self.yp
        sigma, gamma, _, _, xc, yc, nx, ny, tx, ty, _ = panel_solve(xp, yp, a)
        _, Sn, _, Vn, _ = _tangential_and_normal(xp, yp)
        vn = -(math.cos(a) * nx + math.sin(a) * ny)  # = -V_inf·n
        assert np.max(np.abs(Sn @ sigma + gamma * Vn - vn)) < 1e-9


# ==============================================================================
# Valores físicos
# ==============================================================================

class TestPhysics:
    def test_symmetric_airfoil_zero_lift_at_zero_alpha(self):
        s = panel_naca4("0012", alpha=0.0)
        assert s.cl == pytest.approx(0.0, abs=1e-6)
        assert s.cm_c4 == pytest.approx(0.0, abs=1e-6)

    def test_symmetric_airfoil_cp_symmetry_at_zero_alpha(self):
        s = panel_naca4("0012", alpha=0.0, n_points=80)
        assert np.allclose(s.cp, s.cp[::-1], atol=2e-2)

    def test_lift_slope_close_to_thickness_corrected_2pi(self):
        # Pendiente potencial ≈ 2π(1 + 0.77 t/c) para NACA 00xx
        s1 = panel_naca4("0012", alpha=1.0, deg=True)
        s5 = panel_naca4("0012", alpha=5.0, deg=True)
        slope = (s5.cl - s1.cl) / math.radians(4.0)
        expected = 2 * math.pi * (1 + 0.77 * 0.12)
        assert slope == pytest.approx(expected, rel=0.03)

    def test_cl_is_odd_for_symmetric_airfoil(self):
        sp = panel_naca4("0012", alpha=5.0, deg=True)
        sm = panel_naca4("0012", alpha=-5.0, deg=True)
        assert sp.cl == pytest.approx(-sm.cl, abs=1e-6)

    def test_cambered_airfoil_zero_lift_angle(self):
        # NACA 2412: alpha_0 ≈ -2.1° (Abbott & Von Doenhoff)
        s0 = panel_naca4("2412", alpha=0.0)
        slope = 2 * math.pi * (1 + 0.77 * 0.12)
        alpha_0_deg = -math.degrees(s0.cl / slope)
        assert alpha_0_deg == pytest.approx(-2.1, abs=0.3)

    def test_consistent_with_thin_airfoil_theory(self):
        # Entre ambas teorías la diferencia es del orden de la corrección de espesor
        for code in ("0012", "2412", "4412"):
            panel = panel_naca4(code, alpha=4.0, deg=True)
            thin = thin_airfoil_naca4(code, 4.0, deg=True)
            assert panel.cl == pytest.approx(thin.cl, rel=0.15)

    def test_cm_c4_cambered_negative_nose_down(self):
        s = panel_naca4("2412", alpha=2.0, deg=True)
        assert s.cm_c4 < 0
        assert s.cm_c4 == pytest.approx(-0.053, abs=0.02)

    def test_drag_close_to_zero_d_alembert(self):
        for a in (0.0, 4.0, 8.0):
            s = panel_naca4("2412", alpha=a, deg=True)
            assert abs(s.cd) < 0.002

    def test_stagnation_cp_near_one_and_suction_peak(self):
        s = panel_naca4("0012", alpha=5.0, deg=True)
        assert s.cp.max() == pytest.approx(1.0, abs=0.05)
        assert s.cp.min() < -1.0

    def test_circulation_matches_cl(self):
        # Kutta-Joukowski: cl = 2 * Gamma / (V c), Gamma = gamma * perímetro
        af = naca4("2412", n_points=120)
        xp, yp = panel_geometry(af)
        perimeter = float(np.sum(np.hypot(np.diff(xp), np.diff(yp))))
        s = panel_analysis(af, 4.0, deg=True)
        assert 2.0 * s.gamma * perimeter == pytest.approx(s.cl, rel=0.03)

    def test_gamma_sign_follows_lift(self):
        assert panel_naca4("0012", alpha=5.0, deg=True).gamma > 0
        assert panel_naca4("0012", alpha=-5.0, deg=True).gamma < 0

    def test_cp_distribution_is_smooth(self):
        # Sin zigzag panel a panel (problema de la formulación de vórtice constante)
        s = panel_naca4("0012", alpha=4.0, deg=True, n_points=100)
        d2 = np.abs(np.diff(s.cp, 2))
        assert np.max(d2[5:-5]) < 0.5

    def test_naca0012_zero_alpha_pressure_minimum(self):
        # Valor clásico: Cp_min ≈ -0.41 hacia x/c ≈ 0.1
        s = panel_naca4("0012", alpha=0.0, n_points=100)
        assert s.cp.min() == pytest.approx(-0.41, abs=0.05)
        assert s.xc[np.argmin(s.cp)] == pytest.approx(0.11, abs=0.06)

    def test_convergence_with_panel_count(self):
        cl_coarse = panel_naca4("0012", alpha=4.0, deg=True, n_points=40).cl
        cl_fine = panel_naca4("0012", alpha=4.0, deg=True, n_points=160).cl
        cl_mid = panel_naca4("0012", alpha=4.0, deg=True, n_points=100).cl
        assert abs(cl_fine - cl_mid) < abs(cl_fine - cl_coarse) + 1e-3
        assert cl_fine == pytest.approx(cl_mid, abs=0.01)

    def test_naca5_runs_and_has_positive_lift(self):
        s = panel_naca5("23012", alpha=2.0, deg=True)
        assert s.cl > 0.3
        assert s.cm_c4 < 0


# ==============================================================================
# panel_forces
# ==============================================================================

class TestPanelForces:
    def test_zero_cp_gives_zero_forces(self):
        n = 10
        z = np.zeros(n)
        out = panel_forces(z, z, z, z, np.ones(n), np.ones(n) * 0.1, 0.1)
        assert all(v == 0.0 for v in out)

    def test_returns_six_values(self):
        n = 4
        out = panel_forces(np.ones(n), np.ones(n), np.zeros(n),
                           np.zeros(n), np.ones(n), np.ones(n), 0.0)
        assert len(out) == 6

    def test_wind_axes_rotation(self):
        s = panel_naca4("2412", alpha=6.0, deg=True)
        a = s.alpha
        assert s.cl == pytest.approx(s.cn * math.cos(a) - s.ca * math.sin(a))
        assert s.cd == pytest.approx(s.cn * math.sin(a) + s.ca * math.cos(a))

    def test_cm_c4_from_cm_le_and_cn(self):
        s = panel_naca4("2412", alpha=3.0, deg=True)
        assert s.cm_c4 == pytest.approx(s.cm_le + 0.25 * s.cn)


# ==============================================================================
# panel_analysis / PanelState
# ==============================================================================

class TestPanelAnalysis:
    def test_returns_panel_state(self):
        s = panel_analysis(naca4("2412", n_points=60), 0.05)
        assert isinstance(s, PanelState)
        assert s.n_panels == 2 * 60 - 2
        assert s.xc.shape == s.yc.shape == s.cp.shape == s.sigma.shape == (s.n_panels,)
        assert isinstance(s.gamma, float)

    def test_deg_flag_equivalence(self):
        af = naca4("2412", n_points=60)
        s_deg = panel_analysis(af, 5.0, deg=True)
        s_rad = panel_analysis(af, math.radians(5.0))
        assert s_deg.cl == pytest.approx(s_rad.cl)
        assert s_deg.alpha == pytest.approx(math.radians(5.0))

    def test_center_of_pressure_for_symmetric_airfoil_near_quarter_chord(self):
        s = panel_naca4("0012", alpha=4.0, deg=True)
        assert s.center_of_pressure == pytest.approx(0.25, abs=0.02)

    def test_invalid_airfoil(self):
        with pytest.raises(TypeError):
            panel_analysis("2412", 0.0)

    @pytest.mark.parametrize("alpha", ["5", None, [1], True])
    def test_invalid_alpha(self, alpha):
        with pytest.raises(TypeError):
            panel_analysis(naca4("0012", n_points=40), alpha)

    def test_invalid_deg(self):
        with pytest.raises(TypeError):
            panel_analysis(naca4("0012", n_points=40), 1.0, deg=1)

    def test_too_few_panels(self):
        af = Airfoil("x", np.array([0.0, 1.0]), np.array([0.0, 0.0]),
                     np.array([0.0, 1.0]), np.array([0.0, 0.0]))
        with pytest.raises(ValueError):
            panel_analysis(af, 0.0)


# ==============================================================================
# panel_polar
# ==============================================================================

class TestPanelPolar:
    def test_shapes_and_units(self):
        al, cl, cd, cm = panel_polar(naca4("2412", n_points=60), [0, 2, 4], deg=True)
        assert al.shape == cl.shape == cd.shape == cm.shape == (3,)
        assert al == pytest.approx(np.radians([0, 2, 4]))

    def test_cl_monotonic_increasing(self):
        _, cl, _, _ = panel_polar(naca4("2412", n_points=60), np.arange(-4, 9, 2), deg=True)
        assert np.all(np.diff(cl) > 0)

    def test_matches_single_analysis(self):
        af = naca4("2412", n_points=60)
        _, cl, _, cm = panel_polar(af, [3.0], deg=True)
        s = panel_analysis(af, 3.0, deg=True)
        assert cl[0] == pytest.approx(s.cl)
        assert cm[0] == pytest.approx(s.cm_c4)

    def test_invalid_inputs(self):
        af = naca4("0012", n_points=40)
        with pytest.raises(TypeError):
            panel_polar("x", [0.0])
        with pytest.raises(TypeError):
            panel_polar(af, ["a", "b"])
        with pytest.raises(ValueError):
            panel_polar(af, [])
        with pytest.raises(ValueError):
            panel_polar(af, [[1.0, 2.0]])
        with pytest.raises(TypeError):
            panel_polar(af, [1.0], deg="yes")


# ==============================================================================
# panel_pressure
# ==============================================================================

class TestPanelPressure:
    def test_cp_equals_one_minus_vt_squared(self):
        xp, yp = panel_geometry(naca4("0012", n_points=40))
        sigma, gamma, _, _, xc, yc, nx, ny, tx, ty, length = panel_solve(xp, yp, 0.1)
        cp, vt = panel_pressure(sigma, gamma, xp, yp, 0.1, tx, ty)
        assert np.allclose(cp, 1.0 - vt**2)

    def test_zero_solution_gives_free_stream_projection(self):
        xp, yp = panel_geometry(naca4("0012", n_points=40))
        n = len(xp) - 1
        _, _, _, _, _, _, tx, ty, _ = influence_matrix(xp, yp, 0.2)
        _, vt = panel_pressure(np.zeros(n), 0.0, xp, yp, 0.2, tx, ty)
        assert np.allclose(vt, math.cos(0.2) * tx + math.sin(0.2) * ty)
