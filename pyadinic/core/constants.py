# Constantes Físicas e ISA (U.S. Standard Atmosphere 1976 / ICAO)
R_e = 6356766.0         # Radio de la Tierra (m)
g0 = 9.80665            # Gravedad estándar a nivel del mar (m/s^2)
R = 287.0528            # Constante de gas ideal para el aire (J/(kg·K))
gamma_air = 1.4         # Relación de calores específicos

# Valores a Nivel del Mar (Capa 0)
T0 = 288.15             # K
P0 = 101325.0           # Pa
rho0 = 1.225            # kg/m^3
L = -0.0065             # Gradiente térmico Troposfera (K/m)

# Valores en tropopausa (Capa 1)
H_TROPOPAUSE = 11000.0  # m
T11 = 216.65            # K
P11 = 22632.10          # Pa

# Definición de capas ISA estándar hasta la mesopausa (84.852 km geopotencial / ~86 km geométrica)
# Capas: [0: Troposfera, 1: Tropopausa, 2: Estratosfera 1, 3: Estratosfera 2, 4: Estratopausa, 5: Mesosfera 1, 6: Mesosfera 2]
ISA_H_BOUNDS = [0.0, 11000.0, 20000.0, 32000.0, 47000.0, 51000.0, 71000.0, 84852.0]
ISA_LAPSE_RATES = [-0.0065, 0.0, 0.0010, 0.0028, 0.0, -0.0028, -0.0020]
MAX_GEOMETRIC_ALTITUDE = 86000.0
MAX_GEOPOTENTIAL_ALTITUDE = 84852.0