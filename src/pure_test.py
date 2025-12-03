import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI
import numpy.random as rand

compound = "Benzene"
T = 550
P = 101325

# Thermodynamic properties
h_molar = PropsSI("Hmolar", "T", T, "P", P, compound)
s_molar = PropsSI("Smolar", "T", T, "P", P, compound)
d_molar = PropsSI("Dmolar", "T", T, "P", P, compound)


q = PropsSI("Q", "T", T, "P", P, compound)
if q == -1:
    phase = PhaseSI("T", T, "P", P, compound)
    if "liquid" in phase.lower():
        q = 0.0
    elif "gas" in phase.lower() or "vapor" in phase.lower():
        q = 1.0
    else:
        q = np.nan

print(f"Properties of {compound} at T={T} K and P={P} Pa:")
print(f"Hmolar: {h_molar} J/mol")
print(f"Smolar: {s_molar} J/mol-K")
print(f"Dmolar: {1/d_molar} mol/m^3")
print(f"Vapor fraction (Q): {q}")

