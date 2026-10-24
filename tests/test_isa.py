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