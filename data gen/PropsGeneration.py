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
    "temperature": randDB,
    "pressure": randP,
    "mole_frac_benzene": randX,
    "mole_frac_toluene": 1 - randX,
    "enth_mol": np.ones_like(randDB),
    "entr_mol": np.ones_like(randDB),
})

# Start timing
startTime = time.time()

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