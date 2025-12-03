from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

compound1 = "Benzene"
compound2 = "Toluene"

PR = CP.AbstractState("HEOS", f"{compound1}&{compound2}")

PR.set_mole_fractions([0.5, 0.5])

PR.update(CP.PT_INPUTS, 101325, 380)

# print(PR.T())

x_liq = PR.mole_fractions_liquid()
x_vap = PR.mole_fractions_vapor()

print(PR.Q())

print(x_vap)
print(x_liq)

print(PR.hmolar())
print(PR.smolar())