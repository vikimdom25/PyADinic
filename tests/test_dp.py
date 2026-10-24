import pytest
from pyadinic.core import dynamic_pressure

def test_calculo_correcto():
    # Usando densidad del aire a nivel del mar (1.225 kg/m^3) y 10 m/s
    assert dynamic_pressure(1.225, 10) == pytest.approx(61.25)

def test_falla_con_texto():
    with pytest.raises(TypeError):
        dynamic_pressure("1.2", 50)
        
def test_falla_con_densidad_cero_o_negativa():
    with pytest.raises(ValueError):
        dynamic_pressure(0, 50)
    with pytest.raises(ValueError):
        dynamic_pressure(-1.5, 50)

def test_falla_con_velocidad_negativa_o_zero():
    with pytest.raises(ValueError):
        dynamic_pressure(1.225, -10)

    assert dynamic_pressure(1.225, 0) == pytest.approx(0)

def test_falla_con_booleano():
    with pytest.raises(TypeError):
        dynamic_pressure(True, 50)
    with pytest.raises(TypeError):
        dynamic_pressure(1.225, False)
