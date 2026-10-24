import pytest
from pyadinic.core import reynolds_number

def test_calc():
    assert reynolds_number(1,1,1,1) == pytest.approx(1)

@pytest.mark.parametrize("args", [
    (-1, 1, 1, 1),
    (1, -1, 1, 1),
    (1, 1, -1, 1),
    (1, 1, 1, -1),
])
def test_negative_values(args):
    with pytest.raises(ValueError):
        reynolds_number(*args)
                              
@pytest.mark.parametrize("args", [
    ("1", 1, 1, 1),
    (1, "1", 1, 1),
    (1, 1, "1", 1),
    (1, 1, 1, "1"),
])
def test_text_in_values(args):
    with pytest.raises(TypeError):
        reynolds_number(*args)


@pytest.mark.parametrize("args", [
    (True, 1, 1, 1),
    (1,True, 1, 1),
    (1, 1,True, 1),
    (1, 1, 1,True),
])
def test_bool_in_values(args):
    with pytest.raises(TypeError):
        reynolds_number(*args)

@pytest.mark.parametrize("args", [
    (0, 1, 1, 1),
    (1, 0, 1, 1),
    (1, 1, 0, 1),
    (1, 1, 1, 0),
])
def test_zero_values(args):
    with pytest.raises(ValueError):
        reynolds_number(*args)