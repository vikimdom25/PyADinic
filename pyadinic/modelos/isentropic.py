import math
from pyadinic.core.constants import R, gamma_air
from pyadinic.core import speed_of_sound

def total_temperature(temperature, mach, gamma=1.4, rr = False):

    magnitudes = [temperature, mach, gamma]

    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")

    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")
    if mach < 0:
        raise ValueError("Mach Under Zero")
    if temperature <= 0:
        raise ValueError("Temperature Under Zero not allowed")
    if not isinstance(rr,bool):
            raise TypeError("Only booleans allowed")

    if not(rr):
        tt = temperature * (1 + (((gamma - 1)/2)*(mach**2)))
        return tt
    else:
        return (1 + (((gamma - 1)/2)*(mach**2)))

def total_pressure(pressure, mach, gamma=1.4, rr = False):

    magnitudes = [pressure, mach, gamma]
    
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
    
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")
    if mach < 0:
        raise ValueError("Mach Under Zero")
    if pressure <= 0:
        raise ValueError("Pressure Under Zero not allowed")
    if not isinstance(rr,bool):
            raise TypeError("Only booleans allowed")

    if not(rr):
        tp = pressure * ((1 + (((gamma - 1)/2)*(mach**2)))**((gamma)/(gamma - 1)))
        return tp
    else:
        return ((1 + (((gamma - 1)/2)*(mach**2)))**((gamma)/(gamma - 1)))

def total_density(density, mach, gamma=1.4, rr = False):

    magnitudes = [density, mach, gamma]
    
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
    
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")
    if mach < 0:
        raise ValueError("Mach Under Zero")
    if density <= 0:
        raise ValueError("Density Under Zero not allowed")
    if not isinstance(rr,bool):
        raise TypeError("Only booleans allowed")

    if not(rr):
        td = density * ((1 + (((gamma - 1)/2)*(mach**2)))**((1)/(gamma - 1)))
        return td
    else:
        return ((1 + (((gamma - 1)/2)*(mach**2)))**((1)/(gamma - 1)))

def velocity_from_mach(mach, temperature, gamma = gamma_air, gas_constant = R):

    if not isinstance(mach, (int,float)):
        raise TypeError("Error, no numeric args")
    if isinstance(mach, bool):
        raise TypeError("Error, no numeric args")
    if mach < 0:
            raise ValueError("Mach Under Zero")
    vfm = mach * speed_of_sound(temperature, gamma, gas_constant)

    return vfm

def impact_pressure(pressure, mach, gamma):
    ip = total_pressure(pressure, mach, gamma, rr = False) - pressure
    return ip 

def mach_from_pressure_ratio(pressure_ratio, gamma=1.4):

    magnitudes = [pressure_ratio, gamma]
        
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
        
    if gamma <= 1:
            raise ValueError("Gamma out of supported envelope")
    if pressure_ratio < 1:
            raise ValueError("Pressure ratio Under one not allowed")
    
    mfpr = math.sqrt((2/(gamma - 1))*(((pressure_ratio)**((gamma - 1)/gamma)) - 1))
    return mfpr

def mach_from_temperature_ratio(temperature_ratio, gamma=1.4):

    magnitudes = [temperature_ratio, gamma]
        
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
        
    if gamma <= 1:
            raise ValueError("Gamma out of supported envelope")
    if temperature_ratio < 1:
            raise ValueError("Temperature ratio Under one not allowed")
    
    mftr = math.sqrt((2/(gamma - 1))*(temperature_ratio - 1))
    return mftr

def mach_from_density_ratio(density_ratio, gamma=1.4):

    magnitudes = [density_ratio, gamma]
        
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
        
    if gamma <= 1:
            raise ValueError("Gamma out of supported envelope")
    if density_ratio < 1:
            raise ValueError("Density ratio Under one not allowed")
    
    mfdr = math.sqrt((2/(gamma - 1))*((density_ratio ** (gamma - 1)) - 1))
    return mfdr 

def area_ratio_from_mach(mach, gamma=1.4):
    magnitudes = [mach, gamma]
            
    for magnitude in magnitudes:
        if not isinstance(magnitude, (int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude, bool):
            raise TypeError("Error, no numeric args")
            
    if gamma <= 1:
        raise ValueError("Gamma out of supported envelope")
    if mach <= 0:
        raise ValueError("Mach under or equal zero not allowed")

    arfm = (1/mach)*(((2/(gamma + 1))*(1+(((gamma - 1)/2)*(mach**2))))**((gamma + 1)/(2*(gamma - 1))))
    return arfm


def mach_from_area_ratio(area_ratio, gamma=1.4, branch="subsonic"):

    mags = [area_ratio, gamma]

    for mag in mags:
        if not isinstance(mag,(int,float)):
             raise TypeError("No numeric arguments not allowed")
        if isinstance(mag,bool):
             raise TypeError("Booleans not allowed")

    if area_ratio < 1:
        raise ValueError("The area_ratio is out of soported envelope")
    if area_ratio == 1:
        return 1.0
    if gamma <= 1:
         raise ValueError("Gamma out of supported envelope")
    if branch not in ("subsonic", "supersonic"):
        raise ValueError("Invalid branch")
    
    if area_ratio > 1:

        from pyadinic.numerical_methods import solve_brent
        def residual(mach):
            return area_ratio_from_mach(mach, gamma) - area_ratio

        if branch == "subsonic":
            return solve_brent(
                    residual,
                    1e-8,
                    1.0,
                    tolerance=1e-8,
                    max_iterations=100
                )

        if branch == "supersonic":
            return solve_brent(
                residual,
                1.0,
                100.0,
                tolerance=1e-8,
                max_iterations=100
            )
    