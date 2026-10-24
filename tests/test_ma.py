import pytest
from pyadinic.core import mach_number

def test_calculo_correcto():
    assert mach_number(0,320) == pytest.approx(0)

def test_falla_texto():
    with pytest.raises(TypeError):
        mach_number("0",320)

def test_falla_bool():
    with pytest.raises(TypeError):
        mach_number(15,True)

def test_falla_negative_mach():
    with pytest.raises(ValueError):
        mach_number(-15,320)

def test_falla_negative_mach_sound():
    with pytest.raises(ValueError):
        mach_number(15,-320)

def test_falla_sound_zero():
    with pytest.raises(ValueError):
            mach_number(15,0)