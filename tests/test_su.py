import pytest
from pyadinic.modelos import sutherland_model_air, sutherland_model_generic

MU_0 = 1.716e-5
T_0 = 273.15
S = 110.4

def test_sutherland_air_reference_temperature():
    assert sutherland_model_air(T_0) == pytest.approx(MU_0)


@pytest.mark.parametrize("temperature", [
    200,
    250,
    288.15,
    300,
    500,
    1000,
])
def test_sutherland_air_positive_temperature(temperature):
    result = sutherland_model_air(temperature)

    assert result > 0


@pytest.mark.parametrize("temperature", [
    200,
    250,
    288.15,
    300,
    500,
])
def test_air_matches_generic(temperature):
    air = sutherland_model_air(temperature)

    generic = sutherland_model_generic(
        temperature,
        MU_0,
        T_0,
        S
    )

    assert air == pytest.approx(generic)


@pytest.mark.parametrize("temperature", [
    "288.15",
    None,
    [288.15],
])
def test_air_invalid_type(temperature):
    with pytest.raises(TypeError):
        sutherland_model_air(temperature)


def test_air_bool():
    with pytest.raises(TypeError):
        sutherland_model_air(True)


@pytest.mark.parametrize("temperature", [
    0,
    -1,
    -273.15,
])
def test_air_invalid_temperature(temperature):
    with pytest.raises(ValueError):
        sutherland_model_air(temperature)


@pytest.mark.parametrize("args", [
    ("288.15", MU_0, T_0, S),
    (288.15, "1.716e-5", T_0, S),
    (288.15, MU_0, "273.15", S),
    (288.15, MU_0, T_0, "110.4"),
])
def test_generic_invalid_type(args):
    with pytest.raises(TypeError):
        sutherland_model_generic(*args)


@pytest.mark.parametrize("args", [
    (True, MU_0, T_0, S),
    (288.15, True, T_0, S),
    (288.15, MU_0, True, S),
    (288.15, MU_0, T_0, True),
])
def test_generic_bool(args):
    with pytest.raises(TypeError):
        sutherland_model_generic(*args)


@pytest.mark.parametrize("args", [
    (0, MU_0, T_0, S),
    (-1, MU_0, T_0, S),
    (288.15, 0, T_0, S),
    (288.15, -1, T_0, S),
    (288.15, MU_0, 0, S),
    (288.15, MU_0, -1, S),
    (288.15, MU_0, T_0, 0),
    (288.15, MU_0, T_0, -1),
])
def test_generic_invalid_values(args):
    with pytest.raises(ValueError):
        sutherland_model_generic(*args)