import sys, math
# Uso: python mision.py [ruta/a/la/carpeta/que/contiene/pyadinic]
sys.path.insert(0, sys.argv[1] if len(sys.argv) > 1 else ".")
import numpy as np
import pyadinic as pad

KT = 0.514444
FT = 0.3048
def hdr(t): print("\n" + "="*8, t, "="*8)
def chk(name, a, b, tol):
    ok = abs(a-b) <= tol*max(1e-12, abs(b))
    print(f"  [{'OK ' if ok else 'REV'}] {name}: {a:.5g} vs {b:.5g}  (dif {100*(a-b)/b:+.2f}%)")
    return ok

# ---------------- Datos (ver supuestos en el informe) ----------------
W_kg, S, b = 3500.0, 25.96, 15.88      # masa a mitad de misión, área y envergadura (aprox.)
W = W_kg*pad.g0; AR = b*b/S; MAC = S/b; e = 0.85
TAS = 140*KT
h_geo = 10000*FT

# ============ 1. Atmósfera, ISA y ISA+20 ============
hdr("1. Atmosfera ISA y ISA+20 a 10,000 ft")
isa = pad.isa_model(h_geo)
cond = {}
for dT in (0.0, 20.0):
    T = isa.temperature + dT
    rho = isa.pressure/(pad.R*T)
    a = pad.speed_of_sound(T)
    mu = pad.sutherland_model_air(T)
    q = pad.dynamic_pressure(rho, TAS)
    M = pad.tas_to_mach(TAS, a)
    Re = pad.reynolds_number(TAS, MAC, rho, mu)
    eas = pad.tas_to_eas(TAS, rho)
    cas = pad.tas_to_cas(TAS, isa.pressure, a)
    cond[dT] = dict(T=T, rho=rho, a=a, mu=mu, q=q, M=M, Re=Re, eas=eas, cas=cas)
    print(f"  ISA{dT:+.0f}: T={T:.2f} K rho={rho:.4f} a={a:.2f} m/s mu={mu:.3e} q={q:.1f} Pa M={M:.4f} Re={Re:.3e} EAS={eas/KT:.1f} kt CAS={cas/KT:.1f} kt")
c0 = cond[0.0]
chk("rho ISA (isa_model vs p/RT)", isa.density, c0["rho"], 1e-4)
chk("EAS = TAS*sqrt(rho/rho0)", c0["eas"], TAS*math.sqrt(c0["rho"]/pad.rho0), 1e-9)
chk("q = 0.5 rho V^2 = 0.5 rho0 EAS^2", c0["q"], 0.5*pad.rho0*c0["eas"]**2, 1e-9)
cas = c0["cas"]
chk("round-trip CAS->TAS", pad.cas_to_tas(cas, isa.pressure, c0["a"]), TAS, 1e-6)
M_cas = pad.cas_to_mach(cas, isa.pressure); chk("Mach via CAS vs via TAS/a", M_cas, c0["M"], 1e-6)
qc = pad.cas_to_impact_pressure(cas)
chk("impact pressure CAS vs impact_pressure(p,M)", qc, pad.impact_pressure(isa.pressure, c0["M"], 1.4), 1e-4)
chk("CAS->qc->CAS", pad.impact_pressure_to_cas(qc), cas, 1e-8)

# ============ 2. Geometría del perfil ============
hdr("2. Perfil NACA 23015 (representativo; el 208B usa 23017.424 raiz / 23012 punta)")
af = pad.naca5("23015", n_points=120)
xp, yp = pad.panel_geometry(af)
area_num = {}
thk = lambda x: pad.thickness_distribution(x, 0.15)
exact = 2*5*0.15*(0.2969*(2/3) - 0.1260*0.5 - 0.3516*(1/3) + 0.2843*0.25 - 0.1015*0.2)   # integral analítica del espesor
for name, f in (("trapecio", pad.integrate_trapezoidal), ("Simpson", pad.integrate_simpson), ("Simpson3/8", pad.integrate_simpson_38), ("Gauss-Legendre", pad.integrate_gauss_legendre)):
    try:
        area_num[name] = f(thk, 0.0, 1.0, 96 if name != "Simpson3/8" else 96)
    except Exception as ex:
        area_num[name] = float("nan"); print("  ", name, "falló:", ex)
for k, v in area_num.items(): chk(f"area espesor {k} (2*int yt)", 2*v, exact, 2e-3)
perim = float(np.sum(np.hypot(np.diff(xp), np.diff(yp))))
print(f"  perimetro del contorno de paneles = {perim:.4f} c   (>= 2 c, ~2.03 esperable)")

# ============ 3. Polar de sustentación por paneles + thin airfoil ============
hdr("3. Polar 2D (paneles) vs perfiles delgados, spline, derivada y minimos cuadrados con Cholesky")
al = np.arange(-4, 15, 2.0)
_, cl_p, cd_p, cm_p = pad.panel_polar(af, al, deg=True)
thin = [pad.thin_airfoil_naca5("23015", a, deg=True) for a in al]
for a, c, t in list(zip(al, cl_p, thin))[::3]:
    print(f"  alpha={a:5.1f} deg  cl_panel={c:.4f}  cl_thin={t.cl:.4f}")
spl = lambda a: pad.cubic_spline_interpolation(float(a), list(al), list(cl_p))
alpha0_p = pad.solve_brent(lambda a: spl(a), -4.0, 2.0)
alpha0_t = math.degrees(pad.thin_airfoil_alpha_0(naca="23015"))
print(f"  alpha_0: paneles (spline+Brent) = {alpha0_p:.3f} deg | perfiles delgados = {alpha0_t:.3f} deg  (NACA 230xx publicado ~ -1.1 a -1.2 deg)")
slope_spl = pad.numerical_derivative(spl, 4.0, h=1e-3)        # por grado
X = np.column_stack([np.ones_like(al), al]); AtA = (X.T@X).tolist(); Atb = (X.T@cl_p).tolist()
sol_ch = pad.cholesky_solve(AtA, Atb); sol_lu = pad.lu_solve(AtA, Atb); sol_gj = pad.gauss_jordan(AtA, Atb); sol_ge = pad.gaussian_elimination(AtA, Atb)
sol_np = np.polyfit(al, cl_p, 1)[::-1]
print(f"  ajuste lineal cl=c0+c1*alpha: Cholesky={np.round(sol_ch,6)}  LU={np.round(sol_lu,6)}  GaussJordan={np.round(sol_gj,6)}  Gauss={np.round(sol_ge,6)}  numpy={np.round(sol_np,6)}")
a_ch = float(sol_ch[1])*180/math.pi
chk("pendiente: derivada del spline vs minimos cuadrados (Cholesky)", slope_spl, float(sol_ch[1]), 5e-3)
chk("Cholesky vs LU vs Gauss-Jordan vs Gauss vs numpy (pendiente)", max(map(float,[sol_ch[1],sol_lu[1],sol_gj[1],sol_ge[1]])), float(sol_np[1]), 1e-9)
print(f"  dcl/dalpha = {a_ch:.3f} /rad (2pi = {2*math.pi:.3f}; con espesor 2pi(1+0.77 t/c) = {2*math.pi*(1+0.77*0.15):.3f})")
chk("pendiente panel vs 2pi(1+0.77 t/c)", a_ch, 2*math.pi*(1+0.77*0.15), 0.04)
chk("alpha_0 paneles vs thin airfoil", alpha0_p, alpha0_t, 0.25)
cm_thin = pad.thin_airfoil_cm_c4(naca="23015")
print(f"  cm_c/4: paneles = {np.mean(cm_p[2:6]):.4f} | thin airfoil = {cm_thin:.4f}")

# ============ 4. Punto de crucero y 3D ============
hdr("4. Crucero: CL requerido, alpha (Brent), resistencia")
res = {}
for dT in (0.0, 20.0):
    c = cond[dT]
    CL = pad.lift_coefficient(W, c["q"], S)
    k = 1.0/(math.pi*e*AR)
    a2d = pad.solve_brent(lambda a: spl(a) - CL, -4.0, 14.0)             # alpha 2D efectiva [deg]
    a_geo = a2d + math.degrees(CL*k)
    Cf = pad.skin_friction_coefficient(c["Re"], "auto")
    Cf_t = pad.skin_friction_coefficient(c["Re"], "turbulent")
    Cf_s = pad.skin_friction_coefficient_turbulent(c["Re"], "schlichting")
    tc = 0.15; FF = 1 + 2*tc + 60*tc**4
    Cd0 = 2*Cf*FF; Cdi = CL*CL*k; Cd = Cd0 + Cdi
    L = pad.lift_force(CL, c["q"], S); D = pad.drag_force(Cd, c["q"], S)
    res[dT] = dict(CL=CL, a2d=a2d, a_geo=a_geo, Cd0=Cd0, Cdi=Cdi, Cd=Cd, D=D, L=L, Cf=Cf)
    print(f"  ISA{dT:+.0f}: CL={CL:.4f} alpha2D={a2d:.2f} deg alpha_geo={a_geo:.2f} deg | Cf(auto)={Cf:.5f} (turb {Cf_t:.5f}, Schlichting {Cf_s:.5f}) Cd0={Cd0:.5f} Cdi={Cdi:.5f} Cd={Cd:.5f} L/D={pad.lift_to_drag_ratio(CL,Cd):.2f}")
    print(f"           L={L:.0f} N (W={W:.0f} N)  D_ala={D:.0f} N  P_ala=D*V={D*TAS/1000:.1f} kW = {D*TAS/745.7:.0f} hp")
    chk("L = W", L, W, 1e-9)
r0, r20 = res[0.0], res[20.0]
print(f"  Efecto ISA+20 a igual TAS: CL {100*(r20['CL']/r0['CL']-1):+.2f}%  D {100*(r20['D']/r0['D']-1):+.2f}%  (q baja {100*(cond[20.0]['q']/c0['q']-1):+.2f}%)")
chk("Cf turbulento Prandtl vs Schlichting (modelos distintos)", pad.skin_friction_coefficient(c0["Re"],"turbulent"), pad.skin_friction_coefficient_turbulent(c0["Re"],"schlichting"), 0.08)
# ejes viento/cuerpo
cn, ca = pad.wind_to_body_coefficients(r0["CL"], r0["Cd"], r0["a_geo"], deg=True)
cl_b, cd_b = pad.body_to_wind_coefficients(cn, ca, r0["a_geo"], deg=True)
chk("viento->cuerpo->viento (CL)", cl_b, r0["CL"], 1e-12)

# ============ 5. Cp del panel: integracion independiente de Cl ============
hdr("5. Cp(x) a alpha de crucero: Cl por integracion de DCp vs Cl de paneles")
st = pad.panel_analysis(af, r0["a2d"], deg=True)
n = st.n_panels; half = n//2
xc, cp = st.xc, st.cp
def mono(x, y):
    o = np.argsort(x); x, y = x[o], y[o]
    keep = np.concatenate([[True], np.diff(x) > 1e-9]); return x[keep], y[keep]
x_up, cp_up = mono(xc[:half], cp[:half])
x_lo, cp_lo = mono(xc[half:], cp[half:])
cpu = lambda x: pad.cubic_spline_interpolation(x, list(x_up), list(cp_up))
cpl = lambda x: pad.cubic_spline_interpolation(x, list(x_lo), list(cp_lo))
lo, hi = max(x_up[0], x_lo[0]), min(x_up[-1], x_lo[-1])
cl_int = pad.integrate_gauss_legendre(lambda x: cpl(x)-cpu(x), lo, hi, 40)
print(f"  Cl_paneles={st.cl:.4f} Cl_int(DCp dx)={cl_int:.4f}  Cp_min={cp.min():.3f} en x/c={xc[np.argmin(cp)]:.3f}  Cp_max={cp.max():.3f}")
chk("Cl: paneles vs integral de DCp (spline+Gauss-Legendre)", cl_int, st.cl, 0.04)
chk("Cl paneles a alpha_2D vs CL requerido", st.cl, r0["CL"], 5e-3)
Cl_thin = pad.thin_airfoil_naca5("23015", r0["a2d"], deg=True).cl
print(f"  Cl por perfiles delgados al mismo alpha = {Cl_thin:.4f}  (menor: no incluye el espesor)")

# ============ 6. Compresibilidad: Mach critico ============
hdr("6. Mach critico (Prandtl-Glauert + relaciones isentropicas + Brent)")
cp0 = float(cp.min())
def cp_star(M): return (2/(1.4*M*M))*((pad.total_pressure(1.0, M)/pad.total_pressure(1.0, 1.0)) - 1)
def f(M): return cp0/math.sqrt(1-M*M) - cp_star(M)
Mcr = pad.solve_brent(f, 0.2, 0.95)
print(f"  Cp_min(incomp)={cp0:.3f}  M_crit={Mcr:.3f}  (M crucero = {c0['M']:.3f})")
chk("Cp*(M=0.5) vs valor de tablas -2.133", cp_star(0.5), -2.133, 2e-3)
# Vmo 175 KIAS desde el nivel del mar hasta techo
Mmax = 0; hm = 0
for hft in range(0, 25001, 1000):
    s = pad.isa_model(hft*FT)
    cas_v = pad.ias_to_cas(175*KT)
    M = pad.cas_to_mach(cas_v, s.pressure)
    if M > Mmax: Mmax, hm = M, hft
print(f"  M maximo a Vmo=175 KIAS hasta 25,000 ft: {Mmax:.3f} (a {hm} ft)  -> margen sobre M_crit: {Mcr-Mmax:+.3f}")

# ============ 7. Chequeo supersonico (choques + Prandtl-Meyer vs Ackeret) ============
hdr("7. Extra: placa plana M=2, alpha=2 deg (shock-expansion exacta vs Ackeret)")
M1, a_deg = 2.0, 2.0
sh = pad.oblique_shock_relations(M1, theta=a_deg, use_degrees=True)
ex = pad.prandtl_meyer_expansion(M1, a_deg, use_degrees=True)
q_inf = 0.5*1.4*M1*M1                                  # q/p
cl_exact = ((sh.pressure_ratio - ex.pressure_ratio)/q_inf)*math.cos(math.radians(a_deg))
cl_ack = 4*math.radians(a_deg)/math.sqrt(M1*M1-1)
cd_exact = ((sh.pressure_ratio - ex.pressure_ratio)/q_inf)*math.sin(math.radians(a_deg))
chk("Cl: shock-expansion vs Ackeret", cl_exact, cl_ack, 0.02)
print(f"  Cd_onda (exacto) = {cd_exact:.5f} ; Ackeret 4a^2/sqrt(M^2-1) = {4*math.radians(a_deg)**2/math.sqrt(M1*M1-1):.5f}")

# ============ 8. Planeo con RK4 ============
hdr("8. Planeo sin motor (RK4 + ISA variable) vs resultado analitico")
def polar_params(f_extra):
    Cd0 = r0["Cd0"] + f_extra/S; k = 1/(math.pi*e*AR)
    CLs = math.sqrt(Cd0/k); LD = 1/(2*math.sqrt(Cd0*k))
    return Cd0, k, CLs, LD
h0 = 3048.0
print("  f_extra[m2]  Cd0_total  CL*   (L/D)max  L/D numerico(CL/Cd en barrido)  Vbg_TAS[kt]  alcance analitico[km]  analitico+energia[km]  alcance RK4[km]  dif% (puro / con energia)  sink[m/s]")
for f_extra in (0.0, 0.5, 1.0):
    Cd0, k, CLs, LD = polar_params(f_extra)
    grid = np.linspace(0.1, 1.4, 600)
    LDnum = max(c/(Cd0+k*c*c) for c in grid)
    rho_h0 = pad.isa_model(h0).density
    V0 = math.sqrt(2*W/(rho_h0*S*CLs))
    g0_ = -math.atan(1/LD)
    def rhs(t, y):
        V, gam, h, x = y
        rho = pad.isa_model(max(h, 0.0)).density
        qS = 0.5*rho*V*V*S
        L_ = qS*CLs; D_ = qS*(Cd0 + k*CLs**2)
        return [(-D_ - W*math.sin(gam))/(W/pad.g0), (L_ - W*math.cos(gam))/((W/pad.g0)*V), V*math.sin(gam), V*math.cos(gam)]
    t, y = pad.rk4(rhs, (0.0, 1400.0), [V0, g0_, h0, 0.0], 1.0)
    idx = np.where(y[:, 2] <= 0)[0]; i = idx[0] if len(idx) else len(t)-1
    # interpolar al h=0
    if i > 0:
        fr = y[i-1, 2]/(y[i-1, 2]-y[i, 2]); xend = y[i-1, 3] + fr*(y[i, 3]-y[i-1, 3]); tend = t[i-1] + fr*(t[i]-t[i-1])
    else: xend, tend = y[i, 3], t[i]
    dif = 100*(xend - h0*LD)/(h0*LD)
    Vf = math.sqrt(2*W/(pad.rho0*S*CLs))                       # velocidad de equilibrio a nivel del mar con el mismo CL
    x_en = (h0 + (V0**2 - Vf**2)/(2*pad.g0))*LD                 # la desaceleracion al ganar densidad devuelve energia
    dif_en = 100*(xend - x_en)/x_en
    print(f"  {f_extra:8.1f}   {Cd0:.5f}  {CLs:.3f}   {LD:6.2f}       {LDnum:6.2f}                 {V0/KT:6.1f}        {h0*LD/1000:7.2f}             {x_en/1000:7.2f}             {xend/1000:7.2f}      {dif:+.2f} / {dif_en:+.2f}   {V0*math.sin(-g0_):.2f}  (t={tend/60:.1f} min)")
print("  nota: a CL constante L/D es constante, pero la velocidad de equilibrio cae al ganar densidad; esa energia cinetica extra alarga el planeo (columna con energia).")
