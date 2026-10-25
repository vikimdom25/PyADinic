import pytest
from pyadinic.atmosferas import isa_model


def test_isa_ref():
    ATMOSFERA = isa_model(0, False)
    assert ATMOSFERA.temperature == pytest.approx(288.15)
    assert ATMOSFERA.pressure == pytest.approx(101325)
    assert ATMOSFERA.density == pytest.approx(1.225)
    assert ATMOSFERA.speed_of_sound == pytest.approx(340.2939465)


def test_isa_tropopause():
    ATMOSFERA = isa_model(11000, False)
    assert ATMOSFERA.temperature == pytest.approx(216.65)
    assert ATMOSFERA.pressure == pytest.approx(22632.1, abs=0.1)
    assert ATMOSFERA.density == pytest.approx(0.3639, abs=0.0001)
    assert ATMOSFERA.speed_of_sound == pytest.approx(295.07, abs=0.01)


def test_isa_stratosphere_15000():
    ATMOSFERA = isa_model(15000, False)
    assert ATMOSFERA.temperature == pytest.approx(216.65)
    assert ATMOSFERA.pressure == pytest.approx(12044.6, abs=0.1)
    assert ATMOSFERA.density == pytest.approx(0.1937, abs=0.0001)
    assert ATMOSFERA.speed_of_sound == pytest.approx(295.07, abs=0.01)


def test_isa_stratosphere_30000():
    ATMOSFERA = isa_model(30000, False)
    # T = 216.65 + 0.001 * 10000 = 226.65 K
    assert ATMOSFERA.temperature == pytest.approx(226.65)
    assert ATMOSFERA.pressure == pytest.approx(1171.86, rel=1e-3)


def test_isa_stratopause_50000():
    ATMOSFERA = isa_model(50000, False)
    # At 50 km geopotential, isothermal stratopause at 270.65 K
    assert ATMOSFERA.temperature == pytest.approx(270.65)
    assert ATMOSFERA.pressure == pytest.approx(75.94, rel=1e-3)


def test_isa_mesosphere_70000():
    ATMOSFERA = isa_model(70000, False)
    # Mesosphere 1
    assert ATMOSFERA.temperature == pytest.approx(217.45, rel=1e-3)
    assert ATMOSFERA.pressure == pytest.approx(4.634, rel=1e-3)


def test_isa_upper_mesosphere_84000():
    ATMOSFERA = isa_model(84000, False)
    # At 84 km geopotential, temperature ~ 188.65 K, pressure < 1 Pa
    assert ATMOSFERA.temperature == pytest.approx(188.65, abs=0.5)
    assert ATMOSFERA.pressure < 1.0
    assert ATMOSFERA.density > 0


def test_isa_geometric_vs_geopotential():
    # At 10,000 m geometric, geopotential H = (6356766 * 10000) / (6356766 + 10000) ≈ 9984.3 m
    atm_geom = isa_model(10000, is_geometric=True)
    atm_geop = isa_model(9984.30, is_geometric=False)
    assert atm_geom.temperature == pytest.approx(atm_geop.temperature, abs=0.01)
    assert atm_geom.pressure == pytest.approx(atm_geop.pressure, rel=1e-3)


def test_isa_invalid_types():
    with pytest.raises(TypeError):
        isa_model("1000")
    with pytest.raises(TypeError):
        isa_model(True)
    with pytest.raises(TypeError):
        isa_model(10000, is_geometric="yes")


def test_isa_out_of_bounds():
    with pytest.raises(ValueError):
        isa_model(-100)
    with pytest.raises(ValueError):
        isa_model(90000, is_geometric=True)
    with pytest.raises(ValueError):
        isa_model(85000, is_geometric=False)



# ============================================================
# Pruebas B1: Desviación ISA delta_T y density_altitude
# ============================================================

from pyadinic.atmosferas import density_altitude


class TestIsaDeltaT:
    @pytest.mark.parametrize("alt", [0.0, 3048.0, 11000.0, 20000.0])
    def test_delta_t_zero_identical_to_default(self, alt):
        s_default = isa_model(alt)
        s_dt0 = isa_model(alt, delta_T=0.0)
        assert s_default.temperature == s_dt0.temperature
        assert s_default.pressure == s_dt0.pressure
        assert s_default.density == s_dt0.density
        assert s_default.speed_of_sound == s_dt0.speed_of_sound

    def test_isa_10000ft_delta_t_plus_20(self):
        # A 10,000 ft (3048 m geometricos) con delta_T = +20
        alt = 3048.0
        s_dt20 = isa_model(alt, is_geometric=True, delta_T=20.0)
        s_std = isa_model(alt, is_geometric=True)

        # Presion igual a la ISA
        assert s_dt20.pressure == pytest.approx(69694.6, rel=1e-3)
        assert s_dt20.pressure == pytest.approx(s_std.pressure, rel=1e-10)

        # T = ISA + 20 K
        assert s_std.temperature == pytest.approx(268.35, abs=0.05)
        assert s_dt20.temperature == pytest.approx(288.35, abs=0.05)

        # Densidad y velocidad del sonido
        assert s_dt20.density == pytest.approx(0.84202, abs=5e-4)
        assert s_dt20.speed_of_sound == pytest.approx(340.41, abs=0.05)

    @pytest.mark.parametrize("dT", [-40.0, -20.0, -10.0, 10.0, 25.0, 40.0])
    @pytest.mark.parametrize("alt", [1000.0, 5000.0, 12000.0])
    def test_density_ratio_relation(self, alt, dT):
        s_std = isa_model(alt)
        s_dt = isa_model(alt, delta_T=dT)
        # rho(dT) / rho(0) == T_isa / (T_isa + dT)
        expected_ratio = s_std.temperature / (s_std.temperature + dT)
        actual_ratio = s_dt.density / s_std.density
        assert actual_ratio == pytest.approx(expected_ratio, rel=1e-9)

    def test_errors_delta_t(self):
        with pytest.raises(TypeError):
            isa_model(1000, delta_T="20")
        with pytest.raises(TypeError):
            isa_model(1000, delta_T=True)
        # Temperatura resultante <= 0
        with pytest.raises(ValueError):
            isa_model(0, delta_T=-300.0)


class TestDensityAltitude:
    def test_density_altitude_10000ft_plus_20(self):
        # density_altitude(3048, +20) ≈ 3740 m (≈12,271 ft) con tolerancia ±15 m
        h_da = density_altitude(3048.0, delta_T=20.0, is_geometric=True)
        assert h_da == pytest.approx(3740.0, abs=15.0)

    @pytest.mark.parametrize("h", [0.0, 2500.0, 8000.0, 15000.0])
    def test_density_altitude_delta_t_zero(self, h):
        assert density_altitude(h, delta_T=0.0) == pytest.approx(h)

    def test_density_altitude_strictly_increasing_in_delta_t(self):
        h = 4000.0
        dTs = [-15.0, -5.0, 0.0, 10.0, 25.0]
        das = [density_altitude(h, delta_T=dt) for dt in dTs]
        for i in range(len(das) - 1):
            assert das[i + 1] > das[i]

    def test_density_altitude_errors(self):
        with pytest.raises(TypeError):
            density_altitude("3000")
        with pytest.raises(TypeError):
            density_altitude(3000, delta_T="10")
        with pytest.raises(TypeError):
            density_altitude(3000, delta_T=True)
        with pytest.raises(TypeError):
            density_altitude(3000, is_geometric="yes")
        with pytest.raises(ValueError):
            density_altitude(0, delta_T=-300.0)
