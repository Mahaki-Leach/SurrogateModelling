import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI
import numpy.random as rand
import time


# Creating data capable of modelling the interaction parameters between two liquids

compound1 = "Benzene"
compound2 = "Toluene"

size = 1000

# Query bounds from CoolProp
Tmin = PropsSI("Tmin", compound1)           # minimum temperature [K]
Tmax = PropsSI("Tcrit", compound1) - 1.0    # below critical [K]
Pmin = PropsSI("pmin", compound1)           # minimum pressure [Pa]
Pmax = PropsSI("pcrit", compound1) - 1e4    # below critical [Pa]

# Randomized inputs within single-phase region
randT = rand.uniform(Tmin, Tmax, size).round(4)
randP = rand.uniform(Pmin, Pmax, size).round(0)

exportDataFrame = pd.DataFrame({
    # Inputs
    "temperature": randT,
    "pressure": randP,
    "benzene_x": np.nan,
    "toluene_x": np.nan,

	# Outputs
    "benzene_q": np.nan,
    "toluene_q": np.nan,
})


###############

import CoolProp

PR = CoolProp.AbstractState("HEOS", "Benzene&Toluene")

PR.set_mole_fractions([0.5, 0.5])

# PR.set_binary_interaction_double(0, 1, "kij", 0)

# PR.build_phase_envelope("none")

# PE = PR.get_phase_envelope_data()


# for P in range(100_000, 1_000_001, 100_000):
    
#     if P < min(PE.p):
#         # Liquid
#         PR.update(CoolProp.PQ_INPUTS, P, 0)
#     elif P > max(PE.p):
#         # Vapor
#         PR.update(CoolProp.PQ_INPUTS, P, 1)
#     else:
#         # Two-phase

#         # Looping through q values
#         for q in range(1, 99, 1):
#             PR.update(CoolProp.PQ_INPUTS, P, q/100)
#             print(PR.T())
#             # print(f"q = {q}: {PR.mole_fractions_liquid()}, {PR.mole_fractions_vapor()}")

#     # print(PR.mole_fractions_liquid())
#     # print(PR.mole_fractions_vapor())
#     # print(PR.Q())

# # print(PE)

# # Do the dewpoint calculation

# PR.update(CoolProp.PQ_INPUTS, 101325, 0)

# print(PR.T())

# PR.specify_phase(CoolProp.iphase_liquid)

for i in range(150, 600, 10):
    PR.update(CoolProp.PT_INPUTS, 101325, i)
    print(PR.mole_fractions_liquid())
    print(PR.mole_fractions_vapor())
    print(PR.Q())
