import warnings
from .essential_magnitudes import (
    dynamic_pressure,
    mach_number,
    reynolds_number,
    speed_of_sound,
    pressure_coefficient,
    pressure_coefficient_incompressible,
    lift_force,
    lift_coefficient,
    drag_force,
    drag_coefficient,
    pitching_moment,
    pitching_moment_coefficient,
    lift_to_drag_ratio,
    body_to_wind_coefficients,
    wind_to_body_coefficients,
    skin_friction_coefficient_laminar,
    skin_friction_coefficient_turbulent,
    skin_friction_coefficient,
    boundary_layer_thickness,
)

warnings.warn(
    "'pyadinic.core.essencial_magnitudes' has been renamed to 'pyadinic.core.essential_magnitudes'. "
    "Please update your imports.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = [
    "dynamic_pressure",
    "mach_number",
    "reynolds_number",
    "speed_of_sound",
    "pressure_coefficient",
    "pressure_coefficient_incompressible",
    "lift_force",
    "lift_coefficient",
    "drag_force",
    "drag_coefficient",
    "pitching_moment",
    "pitching_moment_coefficient",
    "lift_to_drag_ratio",
    "body_to_wind_coefficients",
    "wind_to_body_coefficients",
    "skin_friction_coefficient_laminar",
    "skin_friction_coefficient_turbulent",
    "skin_friction_coefficient",
    "boundary_layer_thickness",
]
