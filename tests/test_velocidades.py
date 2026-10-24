import pytest
import math
from pyadinic.modelos.velocidades import (
    ias_to_cas,
    cas_to_ias,
    tas_to_eas,
    eas_to_tas,
    tas_to_mach,
    mach_to_tas,
    cas_to_impact_pressure,
    impact_pressure_to_cas,
    cas_to_mach,
    mach_to_cas,
    cas_to_tas,
    tas_to_cas,
    cas_to_eas,
    eas_to_cas,
)


def test_ias_cas():
    ias = 120.0
    cas = ias_to_cas(ias, position_error=2.0, instrument_error=-1.0)
    assert cas == pytest.approx(121.0)
    ias_back = cas_to_ias(cas, position_error=2.0, instrument_error=-1.0)
    assert ias_back == pytest.approx(ias)


def test_eas_tas_roundtrip():
    rho = 0.5
    tas = 200.0
    eas = tas_to_eas(tas, rho)
    assert eas < tas
    tas_back = eas_to_tas(eas, rho)
    assert tas_back == pytest.approx(tas)


def test_eas_tas_sea_level():
    rho0 = 1.225
    tas = 150.0
    assert tas_to_eas(tas, rho0) == pytest.approx(tas)
    assert eas_to_tas(tas, rho0) == pytest.approx(tas)


def test_mach_tas_conversions():
    a = 340.0
    tas = 170.0
    mach = tas_to_mach(tas, a)
    assert mach == pytest.approx(0.5)
    assert mach_to_tas(mach, a) == pytest.approx(tas)


def test_cas_impact_pressure_subsonic_and_supersonic():
    # Subsonic: CAS = 200 m/s
    cas_sub = 200.0
    qc_sub = cas_to_impact_pressure(cas_sub)
    assert impact_pressure_to_cas(qc_sub) == pytest.approx(cas_sub, rel=1e-5)

    # Supersonic: CAS = 500 m/s (> a0 approx 340 m/s)
    cas_sup = 500.0
    qc_sup = cas_to_impact_pressure(cas_sup)
    assert qc_sup > 0
    assert impact_pressure_to_cas(qc_sup) == pytest.approx(cas_sup, rel=1e-4)


def test_cas_tas_roundtrip():
    # At standard sea level, CAS == TAS
    p0 = 101325.0
    a0 = 340.294
    cas = 150.0
    tas = cas_to_tas(cas, p0, a0)
    assert tas == pytest.approx(cas, rel=1e-4)
    cas_back = tas_to_cas(tas, p0, a0)
    assert cas_back == pytest.approx(cas, rel=1e-4)


def test_cas_mach_roundtrip():
    p = 50000.0
    cas = 220.0
    mach = cas_to_mach(cas, p)
    cas_back = mach_to_cas(mach, p)
    assert cas_back == pytest.approx(cas, rel=1e-4)


def test_eas_cas_roundtrip():
    p = 30000.0
    rho = 0.4
    a = 300.0
    cas = 140.0
    eas = cas_to_eas(cas, p, rho, a)
    cas_back = eas_to_cas(eas, p, rho, a)
    assert cas_back == pytest.approx(cas, rel=1e-4)


def test_zero_velocities():
    assert tas_to_eas(0, 1.225) == 0.0
    assert eas_to_tas(0, 1.225) == 0.0
    assert tas_to_mach(0, 340.0) == 0.0
    assert mach_to_tas(0, 340.0) == 0.0
    assert cas_to_impact_pressure(0) == 0.0
    assert impact_pressure_to_cas(0) == 0.0
    assert cas_to_tas(0, 101325.0, 340.0) == 0.0
    assert tas_to_cas(0, 101325.0, 340.0) == 0.0
    assert cas_to_mach(0, 101325.0) == 0.0
    assert mach_to_cas(0, 101325.0) == 0.0


def test_invalid_types():
    with pytest.raises(TypeError):
        tas_to_eas("100", 1.225)
    with pytest.raises(TypeError):
        tas_to_eas(True, 1.225)
    with pytest.raises(TypeError):
        tas_to_mach(100, None)
    with pytest.raises(TypeError):
        cas_to_tas(100, "pressure", 340.0)


def test_invalid_values():
    with pytest.raises(ValueError):
        tas_to_eas(-10, 1.225)
    with pytest.raises(ValueError):
        tas_to_eas(100, -1.225)
    with pytest.raises(ValueError):
        tas_to_mach(100, 0)
    with pytest.raises(ValueError):
        cas_to_impact_pressure(-5)
    with pytest.raises(ValueError):
        ias_to_cas(-10)
