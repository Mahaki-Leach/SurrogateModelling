import math

def binary_phase_exact(z1, K1):
    """
    Compute exact liquid and vapor mole fractions for a binary mixture
    given overall mole fraction z1 (component 1) and K1 (equilibrium constant for component 1).
    """
    z2 = 1 - z1
    x1 = z1 / (z1 + z2 * K1)
    x2 = 1 - x1

    y1 = K1 * x1
    y2 = 1 - y1

    x_liq = [x1, x2]
    y_vap = [y1, y2]

    return x_liq, y_vap

# input > 5.157953123784118, c1_vap 0.9127736640580404, c1_liq 0.1769643194020318

x_liq, y_vap = binary_phase_exact(0.6, 5.157953123784118)

# outputs [0.17168502597190244, 0.8283149740280975] [0.8855433160187316, 0.1144566839812684]

print(x_liq, y_vap)