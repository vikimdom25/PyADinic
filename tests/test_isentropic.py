import pytest

from pyadinic.modelos import (
    total_temperature,
    total_density,
    total_pressure,
    velocity_from_mach,
    impact_pressure,
    mach_from_density_ratio,
    mach_from_pressure_ratio,
    mach_from_temperature_ratio,
    area_ratio_from_mach,
    mach_from_area_ratio
)


# ============================================================
# Total temperature
# ============================================================

def test_total_temperature():
    assert total_temperature(300, 0) == pytest.approx(300)
    assert total_temperature(300, 2) == pytest.approx(540)


def test_total_temperature_ratio():
    assert total_temperature(300, 2, rr=True) == pytest.approx(1.8)


# ============================================================
# Total pressure
# ============================================================

def test_total_pressure():
    assert total_pressure(100000, 0) == pytest.approx(100000)
    assert total_pressure(100000, 1) == pytest.approx(
        189292.9159, rel=1e-5
    )


def test_total_pressure_ratio():
    assert total_pressure(100000, 2, rr=True) == pytest.approx(
        7.82445, rel=1e-5
    )


# ============================================================
# Total density
# ============================================================

def test_total_density():
    assert total_density(1.225, 0) == pytest.approx(1.225)
    assert total_density(1.225, 1) == pytest.approx(
        1.225 * (1.2 ** 2.5)
    )


def test_total_density_ratio():
    assert total_density(1.225, 2, rr=True) == pytest.approx(
        1.8 ** 2.5
    )


# ============================================================
# Velocity from Mach
# ============================================================

def test_velocity_from_mach():
    velocity = velocity_from_mach(1, 288.15)

    assert velocity == pytest.approx(340.3, rel=1e-3)


def test_velocity_from_zero_mach():
    assert velocity_from_mach(0, 288.15) == pytest.approx(0)


# ============================================================
# Impact pressure
# ============================================================

def test_impact_pressure():
    assert impact_pressure(100000, 0, 1.4) == pytest.approx(0)

    assert impact_pressure(100000, 1, 1.4) == pytest.approx(
        89292.9159, rel=1e-5
    )


# ============================================================
# Inverse relations
# ============================================================

@pytest.mark.parametrize("mach", [0, 0.3, 1, 2, 3])
def test_mach_pressure_ratio_round_trip(mach):
    ratio = total_pressure(100000, mach, rr=True)

    recovered_mach = mach_from_pressure_ratio(ratio)

    assert recovered_mach == pytest.approx(mach)


@pytest.mark.parametrize("mach", [0, 0.3, 1, 2, 3])
def test_mach_temperature_ratio_round_trip(mach):
    ratio = total_temperature(300, mach, rr=True)

    recovered_mach = mach_from_temperature_ratio(ratio)

    assert recovered_mach == pytest.approx(mach)


@pytest.mark.parametrize("mach", [0, 0.3, 1, 2, 3])
def test_mach_density_ratio_round_trip(mach):
    ratio = total_density(1.225, mach, rr=True)

    recovered_mach = mach_from_density_ratio(ratio)

    assert recovered_mach == pytest.approx(mach)


# ============================================================
# Zero Mach consistency
# ============================================================

def test_zero_mach_ratios():
    assert total_temperature(300, 0, rr=True) == pytest.approx(1)
    assert total_pressure(100000, 0, rr=True) == pytest.approx(1)
    assert total_density(1.225, 0, rr=True) == pytest.approx(1)


# ============================================================
# Validation
# ============================================================

@pytest.mark.parametrize(
    "function,args",
    [
        (total_temperature, ("300", 1)),
        (total_pressure, ("100000", 1)),
        (total_density, ("1.225", 1)),
        (velocity_from_mach, ("1", 300)),
        (mach_from_pressure_ratio, ("2",)),
        (mach_from_temperature_ratio, ("2",)),
        (mach_from_density_ratio, ("2",)),
    ],
)
def test_invalid_types(function, args):
    with pytest.raises(TypeError):
        function(*args)


@pytest.mark.parametrize(
    "function,args",
    [
        (total_temperature, (300, -1)),
        (total_pressure, (100000, -1)),
        (total_density, (1.225, -1)),
        (velocity_from_mach, (-1, 300)),
        (mach_from_pressure_ratio, (0.9,)),
        (mach_from_temperature_ratio, (0.9,)),
        (mach_from_density_ratio, (0.9,)),
    ],
)
def test_invalid_physical_values(function, args):
    with pytest.raises(ValueError):
        function(*args)


def test_invalid_gamma():
    with pytest.raises(ValueError):
        total_temperature(300, 1, gamma=1)

    with pytest.raises(ValueError):
        total_pressure(100000, 1, gamma=1)

    with pytest.raises(ValueError):
        total_density(1.225, 1, gamma=1)


def test_invalid_rr():
    with pytest.raises(TypeError):
        total_temperature(300, 1, rr=1)

    with pytest.raises(TypeError):
        total_pressure(100000, 1, rr=1)

    with pytest.raises(TypeError):
        total_density(1.225, 1, rr=1)

# ============================================================
# Area ratio from Mach
# ============================================================

def test_area_ratio_from_mach():
    assert area_ratio_from_mach(1) == pytest.approx(1)


def test_area_ratio_subsonic():
    assert area_ratio_from_mach(0.5) == pytest.approx(
        1.33984375, rel=1e-5
    )


def test_area_ratio_supersonic():
    assert area_ratio_from_mach(2) == pytest.approx(
        1.6875, rel=1e-5
    )


@pytest.mark.parametrize("mach", [0.1, 0.3, 0.5, 1, 2, 3, 5])
def test_area_ratio_is_greater_or_equal_to_one(mach):
    assert area_ratio_from_mach(mach) >= 1


@pytest.mark.parametrize("mach", [0.1, 0.5, 1, 2, 3])
def test_area_ratio_with_different_gamma(mach):
    result = area_ratio_from_mach(mach, gamma=1.3)

    assert result >= 1


@pytest.mark.parametrize(
    "mach",
    [
        0,
        -1,
    ],
)
def test_area_ratio_invalid_mach(mach):
    with pytest.raises(ValueError):
        area_ratio_from_mach(mach)


def test_area_ratio_invalid_gamma():
    with pytest.raises(ValueError):
        area_ratio_from_mach(1, gamma=1)

    with pytest.raises(ValueError):
        area_ratio_from_mach(1, gamma=0.5)


@pytest.mark.parametrize(
    "mach,gamma",
    [
        ("1", 1.4),
        (1, "1.4"),
        (True, 1.4),
        (1, True),
    ],
)
def test_area_ratio_invalid_types(mach, gamma):
    with pytest.raises(TypeError):
        area_ratio_from_mach(mach, gamma)

def test_mach_from_area_ratio_unity():
    assert mach_from_area_ratio(1.0) == 1.0


def test_mach_from_area_ratio_subsonic():
    area_ratio = area_ratio_from_mach(0.5)

    mach = mach_from_area_ratio(
        area_ratio,
        branch="subsonic"
    )

    assert mach == pytest.approx(0.5, rel=1e-7)


def test_mach_from_area_ratio_supersonic():
    area_ratio = area_ratio_from_mach(2.0)

    mach = mach_from_area_ratio(
        area_ratio,
        branch="supersonic"
    )

    assert mach == pytest.approx(2.0, rel=1e-7)


def test_mach_from_area_ratio_invalid_area_ratio():
    with pytest.raises(ValueError):
        mach_from_area_ratio(0.9)


def test_mach_from_area_ratio_invalid_gamma():
    with pytest.raises(ValueError):
        mach_from_area_ratio(2.0, gamma=1.0)


def test_mach_from_area_ratio_invalid_branch():
    with pytest.raises(ValueError):
        mach_from_area_ratio(2.0, branch="invalid")


def test_mach_from_area_ratio_non_numeric_area_ratio():
    with pytest.raises(TypeError):
        mach_from_area_ratio("2.0")


def test_mach_from_area_ratio_non_numeric_gamma():
    with pytest.raises(TypeError):
        mach_from_area_ratio(2.0, gamma="1.4")


def test_mach_from_area_ratio_boolean_area_ratio():
    with pytest.raises(TypeError):
        mach_from_area_ratio(True)


def test_mach_from_area_ratio_boolean_gamma():
    with pytest.raises(TypeError):
        mach_from_area_ratio(2.0, gamma=True)

@pytest.mark.parametrize("mach", [0.1, 0.3, 0.5, 0.8])
def test_mach_from_area_ratio_subsonic_inverse(mach):
    area_ratio = area_ratio_from_mach(mach)

    result = mach_from_area_ratio(
        area_ratio,
        branch="subsonic"
    )

    assert result == pytest.approx(mach, rel=1e-7)


@pytest.mark.parametrize("mach", [1.2, 1.5, 2.0, 3.0, 5.0])
def test_mach_from_area_ratio_supersonic_inverse(mach):
    area_ratio = area_ratio_from_mach(mach)

    result = mach_from_area_ratio(
        area_ratio,
        branch="supersonic"
    )

    assert result == pytest.approx(mach, rel=1e-7)