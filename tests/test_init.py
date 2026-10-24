import pyadinic as pad

def test_top_level_exports():
    assert hasattr(pad, "__version__")
    assert pad.__version__ == "0.3.0"
    
    # Atmospheric & Core
    assert callable(pad.isa_model)
    assert callable(pad.dynamic_pressure)
    assert callable(pad.mach_number)
    assert callable(pad.reynolds_number)
    assert callable(pad.speed_of_sound)
    assert callable(pad.pressure_coefficient)
    assert callable(pad.pressure_coefficient_incompressible)
    assert callable(pad.lift_force)
    assert callable(pad.lift_coefficient)
    assert callable(pad.drag_force)
    assert callable(pad.drag_coefficient)
    assert callable(pad.pitching_moment)
    assert callable(pad.pitching_moment_coefficient)
    assert callable(pad.lift_to_drag_ratio)
    assert callable(pad.body_to_wind_coefficients)
    assert callable(pad.wind_to_body_coefficients)
    assert callable(pad.skin_friction_coefficient_laminar)
    assert callable(pad.skin_friction_coefficient_turbulent)
    assert callable(pad.skin_friction_coefficient)
    assert callable(pad.boundary_layer_thickness)
    
    # Panel methods
    assert callable(pad.panel_geometry)
    assert callable(pad.influence_matrix)
    assert callable(pad.panel_solve)
    assert callable(pad.panel_pressure)
    assert callable(pad.panel_forces)
    assert callable(pad.panel_analysis)
    assert callable(pad.panel_polar)
    assert callable(pad.panel_naca4)
    assert callable(pad.panel_naca5)

    # Dataclasses
    assert hasattr(pad, "AtmosphericState")
    assert hasattr(pad, "NormalShockState")
    assert hasattr(pad, "ObliqueShockState")
    assert hasattr(pad, "PrandtlMeyerState")
    assert hasattr(pad, "Airfoil")

    # Models & Airfoils
    assert callable(pad.naca4)
    assert callable(pad.naca5)
    assert callable(pad.parse_naca4)
    assert callable(pad.parse_naca5)
    assert callable(pad.thickness_distribution)
    assert callable(pad.camber_line)
    assert callable(pad.surface_coordinates)
    assert callable(pad.spacing)
    assert callable(pad.monotone_surface)

    # Models & Airspeeds
    assert callable(pad.sutherland_model_air)
    assert callable(pad.total_temperature)
    assert callable(pad.total_pressure)
    assert callable(pad.mach_from_area_ratio)
    assert callable(pad.tas_to_eas)
    assert callable(pad.cas_to_tas)
    assert callable(pad.ias_to_cas)
    assert callable(pad.cas_to_mach)

    # Shocks & Prandtl-Meyer
    assert callable(pad.normal_shock_mach2)
    assert callable(pad.normal_shock_relations)
    assert callable(pad.oblique_shock_beta)
    assert callable(pad.oblique_shock_relations)
    assert callable(pad.prandtl_meyer_expansion)
    assert callable(pad.mach_angle)
    
    # Numerical
    assert callable(pad.solve_brent)
    assert callable(pad.rk4)
    assert callable(pad.integrate_gauss_legendre)
    assert callable(pad.cubic_spline_interpolation)
    assert callable(pad.pchip_interpolation)
    assert callable(pad.gaussian_elimination)
    assert callable(pad.gauss_jordan)
    assert callable(pad.lu_solve)
    assert callable(pad.lu_decomposition)
    assert callable(pad.cholesky_solve)
    assert callable(pad.cholesky_decomposition)
    assert callable(pad.lu)
    assert callable(pad.cholesky)

    # Thin Airfoil Theory
    assert hasattr(pad, "ThinAirfoilState")
    assert hasattr(pad, "PanelState")
    assert hasattr(pad, "PolarState")
    assert callable(pad.thin_airfoil_theory)
    assert callable(pad.thin_airfoil_naca4)
    assert callable(pad.thin_airfoil_naca5)
    assert callable(pad.thin_airfoil_alpha_0)
    assert callable(pad.thin_airfoil_cl)
    assert callable(pad.thin_airfoil_cm_c4)
    assert callable(pad.thin_airfoil_cm_le)
    assert callable(pad.thin_airfoil_center_of_pressure)
    assert callable(pad.thin_airfoil_delta_cp)
    assert callable(pad.fourier_coefficients)

