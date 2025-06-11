import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI
import numpy.random as rand
import time

# Constants
size = 100

# Randomized inputs
randDB = rand.uniform(193.15, 673.15, size).round(4)  # Temperature in K
randP = rand.uniform(1000, 900000, size).round(0)  # Pressure in Pa
randX = rand.uniform(0.2, 0.8, size)  # Mole fraction DB

# Setup DataFrame
exportDataFrame = pd.DataFrame({
    "temperature": randDB,
    "pressure": randP,
    "mole_frac_benzene": randX,
    "mole_frac_toluene": 1 - randX,

    "enth_mol": np.ones_like(randDB),
    "entr_mol": np.ones_like(randDB),
    "vapor_fraction": np.ones_like(randDB),
})

# Mixture string
mixture = "HEOS::Benzene[0.5]&Toluene[0.5]"

# Define temperature and pressure ranges
T_range = np.linspace(350, 1000, 8)  # Kelvin
P_range = np.logspace(3, 6, 6)       # Pascals: 1e3 (1 kPa) to 1e6 (10 bar)

print(f"{'T [K]':>8} {'P [Pa]':>10} {'Q':>8} {'Phase':>25} {'VaporFrac':>12}")
print("="*70)

for T in T_range:
    for P in P_range:
        try:
            q = PropsSI("Q", "T", T, "P", P, mixture)
        except:
            q = float('nan')

        try:
            phase = PhaseSI("T", T, "P", P, mixture)
        except:
            phase = "error"

        # Determine vapor fraction logic
        if np.isnan(q) or q == -1.0:
            if "gas" in phase or "supercritical" in phase:
                vf = 1.0
            elif "liquid" in phase:
                vf = 0.0
            else:
                vf = -1.0
        else:
            vf = q

        print(f"{T:8.2f} {P:10.0f} {q:8.3f} {phase:>25} {vf:12.3f}")

# Start timing
startTime = time.time()

# phase = Phase(f"HEOS::Benzene[{0.5}]&Toluene[{0.5}]", 600, 10000)
# hase = PhaseSI("T", 1000, "P", 1000, f"HEOS::Benzene[{0.5}]&Toluene[{0.5}]")
print(f"Phase at 1000 K and 1 kPa: '{phase}'")

q = PropsSI("Q", "T", 673, "P", 10000, f"HEOS::Benzene[{0.5}]&Toluene[{0.5}]")
print(f"Vapor quality (Q): {q}")


print("Benzene:", PhaseSI("T", 1000, "P", 1000, "HEOS::Benzene"))
print("Toluene:", PhaseSI("T", 1000, "P", 1000, "HEOS::Toluene"))

for row in range(0, size):
    T = randDB[row]
    P = randP[row]
    b_x = randX[row]
    t_x = 1 - b_x

    # Set mixture
    mixture = f"HEOS::Benzene[{b_x}]&Toluene[{t_x}]"

    try:
        h_molar = PropsSI("Hmolar", "T", T, "P", P, mixture)
        s_molar = PropsSI("Smolar", "T", T, "P", P, mixture)

        q = PropsSI("Q", "T", T, "P", P, mixture)

        if(q == -1.0): 
            # Single phase system
            # phase = PhaseSI("T", T, "P", P, mixture)
            phase = PhaseSI("T", 673, "P", 10000, mixture)
            print(f"Phase for row {row}: '{phase}'")
            if phase == "liquid":
                exportDataFrame.loc[row, "vapor_fraction"] = 0.0
            elif phase == "gas":
                exportDataFrame.loc[row, "vapor_fraction"] = 1.0
        else:
            exportDataFrame.loc[row, "vapor_fraction"] = 100
        exportDataFrame.loc[row, "enth_mol"] = h_molar
        exportDataFrame.loc[row, "entr_mol"] = s_molar
        
    except:

        # Fallbacks in case of error or unsupported region
        exportDataFrame.loc[row, "enth_mol"] = -1
        exportDataFrame.loc[row, "entr_mol"] = -1
        

# Finish timing
endTime = time.time()
print(f"Completed in {endTime - startTime:.2f} seconds")

# Save to file
exportDataFrame.to_csv("Saved_Mixture_Data.csv", index=False)