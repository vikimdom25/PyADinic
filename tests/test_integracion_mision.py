import math
import numpy as np
import pytest
import pyadinic as pad

KT = 0.514444
FT = 0.3048

# Datos de mision
W_kg, S, b = 3500.0, 25.96, 15.88
W = W_kg * pad.g0
AR = b * b / S
MAC = S / b
e = 0.85
TAS = 140 * KT
h_geo = 10000 * FT


def assert_chk(a, b, tol, name=""):
    diff = abs(a - b)
    max_val = max(1e-12, abs(b))
    assert diff <= tol * max_val, f"{name}: {a} vs {b} (tol={tol})"


class TestAtmosferaMision:
    @pytest.mark.parametrize("dT", [0.0, 20.0])
    def test_atmosfera_condiciones(self, dT):
        isa = pad.isa_model(h_geo)
        T = isa.temperature + dT
        rho = isa.pressure / (pad.R * T)
        a = pad.speed_of_sound(T)
        mu = pad.sutherland_model_air(T)
        q = pad.dynamic_pressure(rho, TAS)
        M = pad.tas_to_mach(TAS, a)
        Re = pad.reynolds_number(TAS, MAC, rho, mu)
        eas = pad.tas_to_eas(TAS, rho)
        cas = pad.tas_to_cas(TAS, isa.pressure, a)

        if dT == 0.0:
            assert_chk(isa.density, rho, 1e-4, "rho ISA (isa_model vs p/RT)")

        assert_chk(eas, TAS * math.sqrt(rho / pad.rho0), 1e-9, "EAS = TAS*sqrt(rho/rho0)")
        assert_chk(q, 0.5 * pad.rho0 * eas**2, 1e-9, "q = 0.5 rho V^2 = 0.5 rho0 EAS^2")

        # Round-trips
        assert_chk(pad.cas_to_tas(cas, isa.pressure, a), TAS, 1e-6, "round-trip CAS->TAS")
        M_cas = pad.cas_to_mach(cas, isa.pressure)
        assert_chk(M_cas, M, 1e-6, "Mach via CAS vs via TAS/a")
        qc = pad.cas_to_impact_pressure(cas)
        assert_chk(qc, pad.impact_pressure(isa.pressure, M, 1.4), 1e-4, "impact pressure CAS vs impact_pressure(p,M)")
        assert_chk(pad.impact_pressure_to_cas(qc), cas, 1e-8, "CAS->qc->CAS")


class TestGeometriaPerfilMision:
    @pytest.mark.parametrize("method_name,func", [
        ("trapecio", pad.integrate_trapezoidal),
        ("Simpson", pad.integrate_simpson),
        ("Simpson3/8", pad.integrate_simpson_38),
        ("Gauss-Legendre", pad.integrate_gauss_legendre),
    ])
    def test_area_espesor(self, method_name, func):
        thk = lambda x: pad.thickness_distribution(x, 0.15)
        exact = 2 * 5 * 0.15 * (
            0.2969 * (2 / 3) - 0.1260 * 0.5 - 0.3516 * (1 / 3) + 0.2843 * 0.25 - 0.1015 * 0.2
        )
        n_pts = 96
        val = func(thk, 0.0, 1.0, n_pts)
        assert_chk(2 * val, exact, 2e-3, f"area espesor {method_name}")

    def test_perimetro_paneles(self):
        af = pad.naca5("23015", n_points=120)
        xp, yp = pad.panel_geometry(af)
        perim = float(np.sum(np.hypot(np.diff(xp), np.diff(yp))))
        assert perim >= 2.0
        assert perim == pytest.approx(2.0597, abs=0.01)


class TestPolarPanelVsThinAirfoilMision:
    @pytest.fixture(scope="class")
    @classmethod
    def polar_data(cls):
        af = pad.naca5("23015", n_points=120)
        al = np.arange(-4, 15, 2.0)
        _, cl_p, cd_p, cm_p = pad.panel_polar(af, al, deg=True)
        return al, cl_p, cd_p, cm_p

    def test_polar_analisis_lineal(self, polar_data):
        al, cl_p, cd_p, cm_p = polar_data
        spl = lambda a: pad.cubic_spline_interpolation(float(a), list(al), list(cl_p))
        alpha0_p = pad.solve_brent(lambda a: spl(a), -4.0, 2.0)
        alpha0_t = math.degrees(pad.thin_airfoil_alpha_0(naca="23015"))

        slope_spl = pad.numerical_derivative(spl, 4.0, h=1e-3)
        X = np.column_stack([np.ones_like(al), al])
        AtA = (X.T @ X).tolist()
        Atb = (X.T @ cl_p).tolist()

        sol_ch = pad.cholesky_solve(AtA, Atb)
        sol_lu = pad.lu_solve(AtA, Atb)
        sol_gj = pad.gauss_jordan(AtA, Atb)
        sol_ge = pad.gaussian_elimination(AtA, Atb)
        sol_np = np.polyfit(al, cl_p, 1)[::-1]

        a_ch = float(sol_ch[1]) * 180 / math.pi
        assert_chk(slope_spl, float(sol_ch[1]), 5e-3, "pendiente: derivada del spline vs minimos cuadrados (Cholesky)")
        assert_chk(
            max(map(float, [sol_ch[1], sol_lu[1], sol_gj[1], sol_ge[1]])),
            float(sol_np[1]),
            1e-9,
            "Cholesky vs LU vs Gauss-Jordan vs Gauss vs numpy (pendiente)",
        )
        assert_chk(a_ch, 2 * math.pi * (1 + 0.77 * 0.15), 0.04, "pendiente panel vs 2pi(1+0.77 t/c)")
        assert_chk(alpha0_p, alpha0_t, 0.25, "alpha_0 paneles vs thin airfoil")


class TestCruceroY3DMision:
    @pytest.mark.parametrize("dT", [0.0, 20.0])
    def test_equilibrio_crucero(self, dT):
        isa = pad.isa_model(h_geo)
        T = isa.temperature + dT
        rho = isa.pressure / (pad.R * T)
        q = pad.dynamic_pressure(rho, TAS)
        CL = pad.lift_coefficient(W, q, S)
        L = pad.lift_force(CL, q, S)
        assert_chk(L, W, 1e-9, f"L = W para dT={dT}")

    def test_modelos_friccion(self):
        isa = pad.isa_model(h_geo)
        T = isa.temperature
        rho = isa.pressure / (pad.R * T)
        a = pad.speed_of_sound(T)
        mu = pad.sutherland_model_air(T)
        Re = pad.reynolds_number(TAS, MAC, rho, mu)

        cf_prandtl = pad.skin_friction_coefficient(Re, "turbulent")
        cf_schlichting = pad.skin_friction_coefficient_turbulent(Re, "schlichting")
        assert_chk(cf_prandtl, cf_schlichting, 0.08, "Cf turbulento Prandtl vs Schlichting")

    def test_transformacion_ejes(self):
        isa = pad.isa_model(h_geo)
        T = isa.temperature
        rho = isa.pressure / (pad.R * T)
        q = pad.dynamic_pressure(rho, TAS)
        CL = pad.lift_coefficient(W, q, S)
        k = 1.0 / (math.pi * e * AR)

        af = pad.naca5("23015", n_points=120)
        al = np.arange(-4, 15, 2.0)
        _, cl_p, _, _ = pad.panel_polar(af, al, deg=True)
        spl = lambda a: pad.cubic_spline_interpolation(float(a), list(al), list(cl_p))
        a2d = pad.solve_brent(lambda a: spl(a) - CL, -4.0, 14.0)
        a_geo = a2d + math.degrees(CL * k)

        Re = pad.reynolds_number(TAS, MAC, pad.isa_model(h_geo).density, pad.sutherland_model_air(T))
        Cf = pad.skin_friction_coefficient(Re, "auto")
        tc = 0.15
        FF = 1 + 2 * tc + 60 * tc**4
        Cd0 = 2 * Cf * FF
        Cdi = CL * CL * k
        Cd = Cd0 + Cdi

        cn, ca = pad.wind_to_body_coefficients(CL, Cd, a_geo, deg=True)
        cl_b, cd_b = pad.body_to_wind_coefficients(cn, ca, a_geo, deg=True)
        assert_chk(cl_b, CL, 1e-12, "viento->cuerpo->viento (CL)")


class TestCpIntegracionMision:
    def test_cp_integracion(self):
        isa = pad.isa_model(h_geo)
        q = pad.dynamic_pressure(isa.density, TAS)
        CL = pad.lift_coefficient(W, q, S)

        af = pad.naca5("23015", n_points=120)
        al = np.arange(-4, 15, 2.0)
        _, cl_p, _, _ = pad.panel_polar(af, al, deg=True)
        spl = lambda a: pad.cubic_spline_interpolation(float(a), list(al), list(cl_p))
        a2d = pad.solve_brent(lambda a: spl(a) - CL, -4.0, 14.0)

        st = pad.panel_analysis(af, a2d, deg=True)
        half = st.n_panels // 2
        xc, cp = st.xc, st.cp

        x_up, cp_up = pad.monotone_surface(xc[:half], cp[:half])
        x_lo, cp_lo = pad.monotone_surface(xc[half:], cp[half:])

        cpu = lambda x: pad.cubic_spline_interpolation(x, list(x_up), list(cp_up))
        cpl = lambda x: pad.cubic_spline_interpolation(x, list(x_lo), list(cp_lo))

        lo, hi = max(x_up[0], x_lo[0]), min(x_up[-1], x_lo[-1])
        cl_int = pad.integrate_gauss_legendre(lambda x: cpl(x) - cpu(x), lo, hi, 40)

        assert_chk(cl_int, st.cl, 0.04, "Cl: paneles vs integral de DCp")
        assert_chk(st.cl, CL, 5e-3, "Cl paneles a alpha_2D vs CL requerido")


class TestMachCriticoMision:
    def test_mach_critico(self):
        isa = pad.isa_model(h_geo)
        q = pad.dynamic_pressure(isa.density, TAS)
        CL = pad.lift_coefficient(W, q, S)

        af = pad.naca5("23015", n_points=120)
        al = np.arange(-4, 15, 2.0)
        _, cl_p, _, _ = pad.panel_polar(af, al, deg=True)
        spl = lambda a: pad.cubic_spline_interpolation(float(a), list(al), list(cl_p))
        a2d = pad.solve_brent(lambda a: spl(a) - CL, -4.0, 14.0)

        st = pad.panel_analysis(af, a2d, deg=True)
        cp0 = float(st.cp.min())

        def cp_star(M):
            return (2 / (1.4 * M * M)) * ((pad.total_pressure(1.0, M) / pad.total_pressure(1.0, 1.0)) - 1)

        def f(M):
            return cp0 / math.sqrt(1 - M * M) - cp_star(M)

        Mcr = pad.solve_brent(f, 0.2, 0.95)
        assert 0.5 < Mcr < 0.6
        assert_chk(cp_star(0.5), -2.133, 2e-3, "Cp*(M=0.5) vs valor de tablas -2.133")


class TestSupersonicoMision:
    def test_shock_expansion_vs_ackeret(self):
        M1, a_deg = 2.0, 2.0
        sh = pad.oblique_shock_relations(M1, theta=a_deg, use_degrees=True)
        ex = pad.prandtl_meyer_expansion(M1, a_deg, use_degrees=True)
        q_inf = 0.5 * 1.4 * M1 * M1
        cl_exact = ((sh.pressure_ratio - ex.pressure_ratio) / q_inf) * math.cos(math.radians(a_deg))
        cl_ack = 4 * math.radians(a_deg) / math.sqrt(M1 * M1 - 1)

        assert_chk(cl_exact, cl_ack, 0.02, "Cl: shock-expansion vs Ackeret")


class TestPlaneoRK4Mision:
    @pytest.mark.parametrize("f_extra", [0.0, 0.5, 1.0])
    def test_planeo_rk4_alcance(self, f_extra):
        # r0 Cd0
        isa = pad.isa_model(h_geo)
        T = isa.temperature
        rho = isa.pressure / (pad.R * T)
        q = pad.dynamic_pressure(rho, TAS)
        CL = pad.lift_coefficient(W, q, S)
        k = 1.0 / (math.pi * e * AR)
        Re = pad.reynolds_number(TAS, MAC, rho, pad.sutherland_model_air(T))
        Cf = pad.skin_friction_coefficient(Re, "auto")
        tc = 0.15
        FF = 1 + 2 * tc + 60 * tc**4
        Cd0_base = 2 * Cf * FF

        Cd0 = Cd0_base + f_extra / S
        CLs = math.sqrt(Cd0 / k)
        LD = 1.0 / (2.0 * math.sqrt(Cd0 * k))

        h0 = 3048.0
        rho_h0 = pad.isa_model(h0).density
        V0 = math.sqrt(2 * W / (rho_h0 * S * CLs))
        g0_ = -math.atan(1 / LD)

        def rhs(t, y):
            V, gam, h, x = y
            rho_local = pad.isa_model(max(h, 0.0)).density
            qS = 0.5 * rho_local * V * V * S
            L_ = qS * CLs
            D_ = qS * (Cd0 + k * CLs**2)
            return [
                (-D_ - W * math.sin(gam)) / (W / pad.g0),
                (L_ - W * math.cos(gam)) / ((W / pad.g0) * V),
                V * math.sin(gam),
                V * math.cos(gam),
            ]

        t, y = pad.rk4(rhs, (0.0, 1400.0), [V0, g0_, h0, 0.0], 1.0)
        idx = np.where(y[:, 2] <= 0)[0]
        i = idx[0] if len(idx) else len(t) - 1
        if i > 0:
            fr = y[i - 1, 2] / (y[i - 1, 2] - y[i, 2])
            xend = y[i - 1, 3] + fr * (y[i, 3] - y[i - 1, 3])
        else:
            xend = y[i, 3]

        Vf = math.sqrt(2 * W / (pad.rho0 * S * CLs))
        x_en = (h0 + (V0**2 - Vf**2) / (2 * pad.g0)) * LD

        # El alcance RK4 coincide con la prediccion con energia cinetica dentro del 0.1%
        assert_chk(xend, x_en, 0.001, f"alcance RK4 vs analitico+energia para f_extra={f_extra}")
