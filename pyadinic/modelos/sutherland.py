MU_0 = 1.716e-5
T_0 = 273.15
S = 110.4

def sutherland_model_air(temperature):
    if not isinstance(temperature,(int,float)):
        raise TypeError("Error, no numeric args")
    if isinstance(temperature, bool):
        raise TypeError("Error, no numeric args")
    if temperature <= 0:
        raise ValueError("Error, magnitude below or equal zero")
    mu = MU_0 * ((T_0 + S) / (temperature + S)) * ((temperature / T_0) ** (3 / 2))
    return mu

def sutherland_model_generic(temperature,mu_0, t_0, s):

    magnitudes = [temperature,mu_0, t_0, s]
    for magnitude in magnitudes:
        if not isinstance(magnitude,(int,float)):
            raise TypeError("Error, no numeric args")
        if isinstance(magnitude,bool):
            raise TypeError("Error, no numeric args")
        if magnitude <= 0:
            raise ValueError("Error, magnitudes below or equal zero")
    mu = mu_0 * ((t_0 + s) / (temperature + s)) * ((temperature / t_0) ** (3 / 2))
    return mu 