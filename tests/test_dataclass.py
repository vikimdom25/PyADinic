from pyadinic.core import (
    AtmosphericState,
    NormalShockState,
    ObliqueShockState,
    PrandtlMeyerState,
    Airfoil,
    ThinAirfoilState,
    PanelState,
)
import numpy as np
import pytest

estado = AtmosphericState(288.15, 101325, 1.225, 340.3)

def test_state():
    assert estado.temperature == pytest.approx(288.15)
    assert estado.pressure == pytest.approx(101325)
    assert estado.density == pytest.approx(1.225)
    assert estado.speed_of_sound == pytest.approx(340.3)


def test_airfoil_dataclass():
    x_u = np.array([0.0, 0.5, 1.0])
    y_u = np.array([0.0, 0.06, 0.0])
    x_l = np.array([0.0, 0.5, 1.0])
    y_l = np.array([0.0, -0.06, 0.0])

    foil = Airfoil(
        name="Test Foil",
        x_upper=x_u,
        y_upper=y_u,
        x_lower=x_l,
        y_lower=y_l,
    )

    assert foil.name == "Test Foil"
    assert np.array_equal(foil.x_upper, x_u)
    assert np.array_equal(foil.y_upper, y_u)
    assert np.array_equal(foil.x_lower, x_l)
    assert np.array_equal(foil.y_lower, y_l)


def test_thin_airfoil_state_dataclass():
    state = ThinAirfoilState(
        alpha=0.087266,
        cl=0.75,
        cm_c4=-0.05,
        cm_le=-0.2375,
        alpha_0=-0.035,
        lift_slope=6.283185,
        center_of_pressure=0.316667,
        a0=0.122,
        a1=0.08,
        a2=0.015,
    )

    assert state.alpha == pytest.approx(0.087266)
    assert state.cl == pytest.approx(0.75)
    assert state.cm_c4 == pytest.approx(-0.05)
    assert state.cm_le == pytest.approx(-0.2375)
    assert state.alpha_0 == pytest.approx(-0.035)
    assert state.lift_slope == pytest.approx(6.283185)
    assert state.center_of_pressure == pytest.approx(0.316667)
    assert state.a0 == pytest.approx(0.122)
    assert state.a1 == pytest.approx(0.08)
    assert state.a2 == pytest.approx(0.015)


def test_panel_state_dataclass():
    xc = np.array([0.9, 0.5, 0.1])
    yc = np.array([0.01, 0.05, 0.0])
    cp = np.array([0.1, -0.8, 0.9])
    sigma = np.array([0.01, 0.02, -0.03])
    state = PanelState(
        alpha=0.0698,
        cl=0.48,
        cd=0.0003,
        cm_le=-0.12,
        cm_c4=-0.005,
        cn=0.48,
        ca=-0.01,
        center_of_pressure=0.25,
        n_panels=3,
        xc=xc,
        yc=yc,
        cp=cp,
        gamma=0.117,
        sigma=sigma,
    )

    assert state.alpha == pytest.approx(0.0698)
    assert state.cl == pytest.approx(0.48)
    assert state.cd == pytest.approx(0.0003)
    assert state.cm_le == pytest.approx(-0.12)
    assert state.cm_c4 == pytest.approx(-0.005)
    assert state.cn == pytest.approx(0.48)
    assert state.ca == pytest.approx(-0.01)
    assert state.center_of_pressure == pytest.approx(0.25)
    assert state.n_panels == 3
    assert np.array_equal(state.xc, xc)
    assert np.array_equal(state.yc, yc)
    assert np.array_equal(state.cp, cp)
    assert state.gamma == pytest.approx(0.117)
    assert np.array_equal(state.sigma, sigma)
