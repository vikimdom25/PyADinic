from dataclasses import FrozenInstanceError
import math
import numpy as np
import pytest

from pyadinic.core.dataclass import Wing
from pyadinic.modelos.wing import (
    trapezoidal_wing,
    wing_from_area_ar,
    chord_at,
    wing_loading,
)


class TestTrapezoidalWing:
    def test_readme_example(self):
        wing = trapezoidal_wing(span=12.0, root_chord=2.0, tip_chord=1.0, sweep_c4=15.0, deg=True)
        assert isinstance(wing, Wing)
        assert wing.span == 12.0
        assert wing.root_chord == 2.0
        assert wing.tip_chord == 1.0
        assert wing.area == pytest.approx(18.0)
        assert wing.aspect_ratio == pytest.approx(8.0)
        assert wing.taper_ratio == pytest.approx(0.5)
        # MAC = (2/3) * 2 * (1 + 0.5 + 0.25) / (1 + 0.5) = (4/3) * (1.75 / 1.5) = 1.555555...
        assert wing.mac == pytest.approx(1.5555555555555556)
        assert wing.sweep_c4 == pytest.approx(math.radians(15.0))

    def test_sweep_radians(self):
        sweep_rad = 0.35
        wing = trapezoidal_wing(span=10.0, root_chord=2.0, tip_chord=1.0, sweep_c4=sweep_rad, deg=False)
        assert wing.sweep_c4 == pytest.approx(sweep_rad)

    def test_rectangular_wing(self):
        wing = trapezoidal_wing(span=10.0, root_chord=1.5, tip_chord=1.5)
        assert wing.area == pytest.approx(15.0)
        assert wing.aspect_ratio == pytest.approx(10.0 / 1.5)
        assert wing.taper_ratio == pytest.approx(1.0)
        assert wing.mac == pytest.approx(1.5)
        assert wing.sweep_c4 == 0.0

    def test_delta_wing(self):
        wing = trapezoidal_wing(span=8.0, root_chord=4.0, tip_chord=0.0)
        assert wing.area == pytest.approx(16.0)
        assert wing.aspect_ratio == pytest.approx(64.0 / 16.0)
        assert wing.taper_ratio == pytest.approx(0.0)
        assert wing.mac == pytest.approx((2.0 / 3.0) * 4.0)

    def test_immutability(self):
        wing = trapezoidal_wing(span=10.0, root_chord=2.0, tip_chord=1.0)
        with pytest.raises(FrozenInstanceError):
            wing.span = 15.0
        with pytest.raises(FrozenInstanceError):
            wing.area = 20.0

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            trapezoidal_wing(span=-10.0, root_chord=2.0, tip_chord=1.0)
        with pytest.raises(ValueError):
            trapezoidal_wing(span=10.0, root_chord=0.0, tip_chord=1.0)
        with pytest.raises(ValueError):
            trapezoidal_wing(span=10.0, root_chord=2.0, tip_chord=-0.5)
        with pytest.raises(TypeError):
            trapezoidal_wing(span="10", root_chord=2.0, tip_chord=1.0)
        with pytest.raises(TypeError):
            trapezoidal_wing(span=10.0, root_chord=True, tip_chord=1.0)
        with pytest.raises(TypeError):
            trapezoidal_wing(span=10.0, root_chord=2.0, tip_chord=1.0, deg="yes")


class TestWingFromAreaAr:
    def test_round_trip_with_trapezoidal(self):
        w_orig = trapezoidal_wing(span=12.0, root_chord=2.0, tip_chord=1.0, sweep_c4=15.0, deg=True)
        w_new = wing_from_area_ar(
            area=w_orig.area,
            aspect_ratio=w_orig.aspect_ratio,
            taper_ratio=w_orig.taper_ratio,
            sweep_c4=15.0,
            deg=True,
        )
        assert w_new.span == pytest.approx(w_orig.span)
        assert w_new.root_chord == pytest.approx(w_orig.root_chord)
        assert w_new.tip_chord == pytest.approx(w_orig.tip_chord)
        assert w_new.mac == pytest.approx(w_orig.mac)
        assert w_new.sweep_c4 == pytest.approx(w_orig.sweep_c4)

    def test_rectangular_from_area_ar(self):
        w = wing_from_area_ar(area=20.0, aspect_ratio=5.0, taper_ratio=1.0)
        assert w.span == pytest.approx(10.0)
        assert w.root_chord == pytest.approx(2.0)
        assert w.tip_chord == pytest.approx(2.0)
        assert w.mac == pytest.approx(2.0)

    def test_invalid_arguments(self):
        with pytest.raises(ValueError):
            wing_from_area_ar(area=-10.0, aspect_ratio=6.0)
        with pytest.raises(ValueError):
            wing_from_area_ar(area=10.0, aspect_ratio=0.0)
        with pytest.raises(ValueError):
            wing_from_area_ar(area=10.0, aspect_ratio=6.0, taper_ratio=-0.1)
        with pytest.raises(TypeError):
            wing_from_area_ar(area=True, aspect_ratio=6.0)
        with pytest.raises(TypeError):
            wing_from_area_ar(area=10.0, aspect_ratio="6.0")


class TestChordAt:
    @pytest.fixture
    def sample_wing(self):
        return trapezoidal_wing(span=12.0, root_chord=2.0, tip_chord=1.0)

    def test_root_chord(self, sample_wing):
        assert chord_at(sample_wing, y=0.0) == pytest.approx(2.0)

    def test_tip_chord_positive_and_negative(self, sample_wing):
        assert chord_at(sample_wing, y=6.0) == pytest.approx(1.0)
        assert chord_at(sample_wing, y=-6.0) == pytest.approx(1.0)

    def test_mid_semi_span_readme_example(self, sample_wing):
        # y = 3.0 m (semienvergadura es 6.0 m -> eta = 0.5)
        # c = 2.0 + 0.5 * (1.0 - 2.0) = 1.5 m
        assert chord_at(sample_wing, y=3.0) == pytest.approx(1.5)
        assert chord_at(sample_wing, y=-3.0) == pytest.approx(1.5)

    def test_vector_input(self, sample_wing):
        y_vec = [0.0, 3.0, 6.0]
        c_vec = chord_at(sample_wing, y_vec)
        assert isinstance(c_vec, np.ndarray)
        assert len(c_vec) == 3
        assert np.allclose(c_vec, [2.0, 1.5, 1.0])

    def test_out_of_bounds(self, sample_wing):
        with pytest.raises(ValueError):
            chord_at(sample_wing, y=6.1)
        with pytest.raises(ValueError):
            chord_at(sample_wing, y=-6.1)

    def test_invalid_types(self, sample_wing):
        with pytest.raises(TypeError):
            chord_at("not_a_wing", y=2.0)
        with pytest.raises(TypeError):
            chord_at(sample_wing, y=True)
        with pytest.raises(TypeError):
            chord_at(sample_wing, y="3.0")


class TestWingLoading:
    def test_with_wing_instance_readme_example(self):
        wing = trapezoidal_wing(span=12.0, root_chord=2.0, tip_chord=1.0)
        # S = 18 m², W = 25,000 N -> W/S = 25000 / 18 ≈ 1388.89 N/m²
        ws = wing_loading(weight=25000.0, wing=wing)
        assert ws == pytest.approx(25000.0 / 18.0)

    def test_with_float_area(self):
        ws = wing_loading(weight=20000.0, wing=16.0)
        assert ws == pytest.approx(1250.0)

    def test_invalid_arguments(self):
        wing = trapezoidal_wing(span=10.0, root_chord=2.0, tip_chord=1.0)
        with pytest.raises(ValueError):
            wing_loading(weight=-1000.0, wing=wing)
        with pytest.raises(ValueError):
            wing_loading(weight=1000.0, wing=-5.0)
        with pytest.raises(ValueError):
            wing_loading(weight=1000.0, wing=0.0)
        with pytest.raises(TypeError):
            wing_loading(weight=True, wing=wing)
        with pytest.raises(TypeError):
            wing_loading(weight=1000.0, wing="15.0")
        with pytest.raises(TypeError):
            wing_loading(weight=1000.0, wing=None)
