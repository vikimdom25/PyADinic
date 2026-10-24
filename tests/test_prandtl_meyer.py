import math
import pytest
from pyadinic.modelos.prandtl_meyer import (
    mach_angle,
    max_prandtl_meyer_angle,
    prandtl_meyer_function,
    mach_from_prandtl_meyer,
    prandtl_meyer_expansion,
)


def test_mach_angle():
    # M = 1 -> mu = 90 deg
    assert mach_angle(1.0, use_degrees=True) == pytest.approx(90.0)
    # M = 2 -> mu = 30 deg
    assert mach_angle(2.0, use_degrees=True) == pytest.approx(30.0)
    assert mach_angle(2.0, use_degrees=False) == pytest.approx(math.pi / 6.0)


def test_prandtl_meyer_sonic():
    # At M = 1.0, nu = 0
    assert prandtl_meyer_function(1.0) == 0.0


def test_prandtl_meyer_known_reference():
    # Standard gas dynamics reference: at M = 2.0 and gamma = 1.4, nu approx 26.38 deg
    nu_deg = prandtl_meyer_function(2.0, use_degrees=True)
    assert nu_deg == pytest.approx(26.38, abs=0.05)


def test_max_prandtl_meyer_angle():
    # Theoretical maximum angle for gamma = 1.4 is approx 130.45 deg
    nu_max_deg = max_prandtl_meyer_angle(1.4, use_degrees=True)
    assert nu_max_deg == pytest.approx(130.45, abs=0.1)


def test_mach_from_prandtl_meyer_roundtrip():
    mach_initial = 2.5
    nu = prandtl_meyer_function(mach_initial)
    mach_back = mach_from_prandtl_meyer(nu)
    assert mach_back == pytest.approx(mach_initial, rel=1e-5)


def test_prandtl_meyer_expansion_physics():
    # Expansion from M1 = 1.5 through turn of theta = 10 deg
    state = prandtl_meyer_expansion(1.5, 10.0, use_degrees=True)

    assert state.mach_1 == 1.5
    assert state.mach_2 > 1.5
    assert state.turn_angle == 10.0
    # In an expansion wave, Mach increases while static pressure, temperature, and density drop
    assert state.pressure_ratio < 1.0
    assert state.temperature_ratio < 1.0
    assert state.density_ratio < 1.0
    assert state.nu_2 == pytest.approx(state.nu_1 + 10.0)
    assert state.mu_2 < state.mu_1


def test_prandtl_meyer_invalid():
    with pytest.raises(ValueError):
        mach_angle(0.5)  # Subsonic
    with pytest.raises(ValueError):
        prandtl_meyer_function(0.8)
    with pytest.raises(ValueError):
        mach_from_prandtl_meyer(-5.0)
    with pytest.raises(ValueError):
        mach_from_prandtl_meyer(150.0, use_degrees=True)  # Exceeds max angle
    with pytest.raises(TypeError):
        mach_angle("2.0")
    with pytest.raises(TypeError):
        prandtl_meyer_expansion(2.0, 10.0, use_degrees="false")
