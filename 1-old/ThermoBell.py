import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI
import numpy.random as rand
import time
from thermo import Mixture

# Constants
size = 10000

# Randomized inputs
randDB = rand.uniform(193.15, 673.15, size).round(4)  # Temperature in K
randP = rand.uniform(1000, 900000, size).round(0)  # Pressure in Pa
randX = rand.uniform(0.2, 0.8, size)  # Mole fraction DB

# Setup DataFrame
exportDataFrame = pd.DataFrame({
    "T": randDB,
    "P": randP,
    "mole_frac_benzene": randX,
    "mole_frac_toluene": 1 - randX,
    "enth_mol": np.ones_like(randDB),
    "entr_mol": np.ones_like(randDB),
    "h_benzene": np.ones_like(randDB),
    "h_toluene": np.ones_like(randDB),
    "s_benzene": np.ones_like(randDB),
    "s_toluene": np.ones_like(randDB),

})

startTime = time.time()

for row in range(0, size):
    T = randDB[row]
    P = randP[row]
    b_x = randX[row]
    t_x = 1 - b_x

    try:
        # Create thermo mixture
        mix = Mixture(["benzene", "toluene"], zs=[b_x, t_x])

        # Total mixture properties
        h_molar = mix.Hmolar
        s_molar = mix.Smolar

        # Partial molar properties
        h_partials = mix.Hmolar_partials
        s_partials = mix.Smolar_partials

        exportDataFrame.loc[row, "T"] = h_molar
        exportDataFrame.loc[row, "P"] = s_molar

        exportDataFrame.loc[row, "mole_frac_benzene"] = randX
        exportDataFrame.loc[row, "mole_frac_toluene"] = 1 - randX

        exportDataFrame.loc[row, "enth_mol"] = h_molar
        exportDataFrame.loc[row, "entr_mol"] = s_molar

        exportDataFrame.loc[row, "h_benzene"] = h_partials[0]
        exportDataFrame.loc[row, "h_toluene"] = h_partials[1]
        exportDataFrame.loc[row, "s_benzene"] = s_partials[0]
        exportDataFrame.loc[row, "s_toluene"] = s_partials[1]

        # Estimate phase (if needed)
        exportDataFrame.loc[row, "q"] = 1.0 if mix.phase == 'g' else 0.0

    except Exception as e:
        print(e)
        # exportDataFrame.loc[row, ["enth_mol", "entr_mol",
        #                         "h_benzene", "s_benzene",
        #                         "h_toluene", "s_toluene"]] = -1


# Finish timing
endTime = time.time()
print(f"Completed in {endTime - startTime:.2f} seconds")

# Save to file
exportDataFrame.to_csv("mixv2.csv", index=False)