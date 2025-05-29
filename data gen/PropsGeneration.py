import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
import numpy.random as rand
import time

# Constants
size = 10000

# Randomized inputs
randDB = rand.uniform(193.15, 673.15, size).round(4)  # Temperature in K
randP = rand.uniform(1000, 900000, size).round(2)  # Pressure in Pa
randX = rand.uniform(0.2, 0.8, size)  # Mole fraction DB

# Setup DataFrame
exportDataFrame = pd.DataFrame({
    "temperature (K)": randDB,
    "pressure (Pa)": randP,
    "mole_frac_benzene": randX,
    "mole_frac_toluene": 1 - randX,
    "molar_density (mol/m3)": np.ones_like(randDB),
    "mass_density (kg/m3)": np.ones_like(randDB),
    "molar_enthalpy (J/mol)": np.ones_like(randDB),
    "mass_enthalpy (J/kg)": np.ones_like(randDB),
    "molar_entropy (J/mol/K)": np.ones_like(randDB),
    "mass_entropy (J/kg/K)": np.ones_like(randDB),
})

# Set mixture
mixture = "HEOS::Benzene[0.5]&Toluene[0.5]"

# Start timing
startTime = time.time()

for row in range(0, size):
    T = randDB[row]
    P = randP[row]

    try:
        rho_molar = PropsSI("Dmolar", "T", T, "P", P, mixture)  # mol/m3
        rho_mass = PropsSI("D", "T", T, "P", P, mixture)         # kg/m3
        h_molar = PropsSI("Hmolar", "T", T, "P", P, mixture)     # J/mol
        h_mass = PropsSI("H", "T", T, "P", P, mixture)           # J/kg
        s_molar = PropsSI("Smolar", "T", T, "P", P, mixture)     # J/mol/K
        s_mass = PropsSI("S", "T", T, "P", P, mixture)           # J/kg/K

        exportDataFrame.loc[row, "molar_density (mol/m3)"] = rho_molar
        exportDataFrame.loc[row, "mass_density (kg/m3)"] = rho_mass
        exportDataFrame.loc[row, "molar_enthalpy (J/mol)"] = h_molar
        exportDataFrame.loc[row, "mass_enthalpy (J/kg)"] = h_mass
        exportDataFrame.loc[row, "molar_entropy (J/mol/K)"] = s_molar
        exportDataFrame.loc[row, "mass_entropy (J/kg/K)"] = s_mass

    except:
        # Fallbacks in case of error or unsupported region
        exportDataFrame.loc[row, "molar_density (mol/m3)"] = -1
        exportDataFrame.loc[row, "mass_density (kg/m3)"] = -1
        exportDataFrame.loc[row, "molar_enthalpy (J/mol)"] = -1
        exportDataFrame.loc[row, "mass_enthalpy (J/kg)"] = -1
        exportDataFrame.loc[row, "molar_entropy (J/mol/K)"] = -1
        exportDataFrame.loc[row, "mass_entropy (J/kg/K)"] = -1

# Finish timing
endTime = time.time()
print(f"Completed in {endTime - startTime:.2f} seconds")

# Save to file
exportDataFrame.to_csv("Saved_Mixture_Data.csv", index=False)