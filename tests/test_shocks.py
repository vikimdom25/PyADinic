import math
import pytest
from pyadinic.modelos.shocks import (
    normal_shock_mach2,
    normal_shock_pressure_ratio,
    normal_shock_density_ratio,
    normal_shock_temperature_ratio,
    normal_shock_total_pressure_ratio,
    rayleigh_pitot_ratio,
    normal_shock_relations,
    mach1_from_normal_shock_pressure_ratio,
    mach1_from_rayleigh_pitot,
    theta_from_beta_mach,
    max_deflection_angle,
    oblique_shock_beta,
    oblique_shock_relations,
)


# =============================================================================
# PRUEBAS DE ONDAS DE CHOQUE NORMAL
# =============================================================================

def test_normal_shock_sonic_limit():
    # At M1 = 1.0, there is no jump across the shock
    assert normal_shock_mach2(1.0) == pytest.approx(1.0)
    assert normal_shock_pressure_ratio(1.0) == pytest.approx(1.0)
    assert normal_shock_density_ratio(1.0) == pytest.approx(1.0)
    assert normal_shock_temperature_ratio(1.0) == pytest.approx(1.0)
    assert normal_shock_total_pressure_ratio(1.0) == pytest.approx(1.0)


def test_normal_shock_m2_reference():
    # Canonical textbook values for M1 = 2.0 and gamma = 1.4 (Anderson / NACA 1135)
    m1 = 2.0
    assert normal_shock_mach2(m1) == pytest.approx(math.sqrt(1 / 3), rel=1e-5)
    assert normal_shock_pressure_ratio(m1) == pytest.approx(4.5, rel=1e-5)
    assert normal_shock_density_ratio(m1) == pytest.approx(8 / 3, rel=1e-5)
    assert normal_shock_temperature_ratio(m1) == pytest.approx(1.6875, rel=1e-5)
    assert normal_shock_total_pressure_ratio(m1) == pytest.approx(0.72087, rel=1e-4)
    assert rayleigh_pitot_ratio(m1) == pytest.approx(5.6404, rel=1e-4)


def test_normal_shock_relations_dataclass():
    state = normal_shock_relations(2.0)
    assert state.mach_1 == 2.0
    assert state.mach_2 == pytest.approx(math.sqrt(1 / 3), rel=1e-5)
    assert state.pressure_ratio == pytest.approx(4.5, rel=1e-5)
    assert state.total_pressure_ratio < 1.0


def test_normal_shock_inversions():
    m1 = 2.5
    p2_p1 = normal_shock_pressure_ratio(m1)
    m1_calc = mach1_from_normal_shock_pressure_ratio(p2_p1)
    assert m1_calc == pytest.approx(m1, rel=1e-5)

    p02_p1 = rayleigh_pitot_ratio(m1)
    m1_pitot = mach1_from_rayleigh_pitot(p02_p1)
    assert m1_pitot == pytest.approx(m1, rel=1e-5)


def test_normal_shock_invalid():
    with pytest.raises(ValueError):
        normal_shock_mach2(0.8)  # Subsonic
    with pytest.raises(ValueError):
        normal_shock_pressure_ratio(1.5, gamma=0.9)
    with pytest.raises(TypeError):
        normal_shock_mach2("2.0")
    with pytest.raises(TypeError):
        normal_shock_mach2(True)


# =============================================================================
# PRUEBAS DE ONDAS DE CHOQUE OBLICUAS
# =============================================================================

def test_theta_beta_mach():
    # For M1 = 2.0, beta = 39.314 deg -> theta approx 10 deg (weak shock)
    theta = theta_from_beta_mach(39.314, 2.0, use_degrees=True)
    assert theta == pytest.approx(10.0, abs=0.05)


def test_max_deflection_angle():
    # For M = 2.0, gamma = 1.4: theta_max approx 22.97 deg, beta_max approx 64.67 deg
    theta_max, beta_max = max_deflection_angle(2.0, use_degrees=True)
    assert theta_max == pytest.approx(22.97, abs=0.1)
    assert beta_max == pytest.approx(64.67, abs=0.1)


def test_oblique_shock_beta_branches():
    m1 = 2.0
    theta_deg = 10.0

    beta_weak = oblique_shock_beta(m1, theta_deg, branch="weak", use_degrees=True)
    assert beta_weak == pytest.approx(39.31, abs=0.1)

    beta_strong = oblique_shock_beta(m1, theta_deg, branch="strong", use_degrees=True)
    assert beta_strong == pytest.approx(83.70, abs=0.1)
    assert beta_strong > beta_weak


def test_oblique_shock_relations():
    # Test complete state for M1 = 2.0, theta = 10 deg
    state_weak = oblique_shock_relations(2.0, theta=10.0, branch="weak", use_degrees=True)
    assert state_weak.mach_1 == 2.0
    assert state_weak.deflection_angle == pytest.approx(10.0)
    assert state_weak.wave_angle == pytest.approx(39.31, abs=0.1)
    assert state_weak.mach_2 > 1.0  # Downstream Mach is supersonic for weak shock
    assert state_weak.pressure_ratio > 1.0

    state_strong = oblique_shock_relations(2.0, theta=10.0, branch="strong", use_degrees=True)
    assert state_strong.mach_2 < 1.0  # Downstream Mach is subsonic for strong shock


def test_oblique_shock_from_beta():
    state = oblique_shock_relations(2.0, beta=45.0, use_degrees=True)
    assert state.wave_angle == 45.0
    assert state.normal_mach_1 == pytest.approx(2.0 * math.sin(math.radians(45.0)))


def test_oblique_shock_detached():
    # For M = 2.0, theta_max is approx 23 deg. An angle of 30 deg must detach.
    with pytest.raises(ValueError):
        oblique_shock_beta(2.0, 30.0, use_degrees=True)


def test_oblique_shock_invalid():
    with pytest.raises(ValueError):
        oblique_shock_beta(0.9, 5.0, use_degrees=True)
    with pytest.raises(ValueError):
        oblique_shock_relations(2.0)  # Neither beta nor theta
    with pytest.raises(ValueError):
        oblique_shock_relations(2.0, beta=40.0, theta=10.0)  # Both specified
    with pytest.raises(TypeError):
        oblique_shock_beta(2.0, 10.0, use_degrees="yes")
