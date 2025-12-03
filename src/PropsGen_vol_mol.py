import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI, PhaseSI
import numpy.random as rand
import time

compound = "Toluene"
size = 1000

# Query bounds from CoolProp
Tmin = PropsSI("Tmin", compound)           # minimum temperature [K]
Tmax = PropsSI("Tcrit", compound) - 1.0    # below critical [K]
Pmin = PropsSI("pmin", compound)           # minimum pressure [Pa]
Pmax = PropsSI("pcrit", compound) - 1e4    # below critical [Pa]

# Randomized inputs within single-phase region
randT = rand.uniform(Tmin, Tmax, size).round(4)
randP = rand.uniform(Pmin, Pmax, size).round(0)

exportDataFrame = pd.DataFrame({
    # Inputs
    "temperature": randT,
    "pressure": randP,
    "q": np.nan,

	# Outputs
    "enth_mol": np.nan,
    "entr_mol": np.nan,
    "vol_mol": np.nan,
})

startTime = time.time()

for row in range(size):
    T = randT[row]
    P = randP[row]

    try:
        # Thermodynamic properties
        h_molar = PropsSI("Hmolar", "T", T, "P", P, compound)
        s_molar = PropsSI("Smolar", "T", T, "P", P, compound)
        d_molar = PropsSI("Dmolar", "T", T, "P", P, compound)

        # Determine vapor fraction (Q)
        try:
            q = PropsSI("Q", "T", T, "P", P, compound)
            if q == -1:
                phase = PhaseSI("T", T, "P", P, compound)
                if "liquid" in phase.lower():
                    q = 0.0
                elif "gas" in phase.lower() or "vapor" in phase.lower():
                    q = 1.0
                else:
                    q = np.nan  # unknown/supercritical
        except Exception:
            q = np.nan

        # Store results
        exportDataFrame.loc[row, ["enth_mol", "entr_mol", "vol_mol", "q"]] = [
            h_molar,
            s_molar,
            1 / d_molar,
            q,
        ]

    except Exception as e:
        print(f"Error at row {row}: {e}")

# Two-phase (VLE) data generation
T_min = PropsSI("Ttriple", compound)
T_max = 0.95 * PropsSI("Tcrit", compound)

randT = np.random.uniform(T_min, T_max, size)
randQ = np.random.uniform(0.01, 0.99, size)

data_vle = pd.DataFrame(columns=["temperature", "pressure", "enth_mol", "entr_mol", "vol_mol", "q"])

for i in range(size):
    T = randT[i]
    Q = randQ[i]

    try:
        Psat = PropsSI("P", "T", T, "Q", 0, compound)

        h = PropsSI("Hmolar", "T", T, "Q", Q, compound)
        s = PropsSI("Smolar", "T", T, "Q", Q, compound)
        d = PropsSI("Dmolar", "T", T, "Q", Q, compound)

        data_vle.loc[i] = [T, Psat, h, s, 1 / d, Q]

    except Exception as e:
        print(f"Error at row {i}: {e}")

endTime = time.time()
print(f"Completed in {endTime - startTime:.2f} seconds")

# Combine single-phase + two-phase data
combinedDataFrame = pd.concat([exportDataFrame, data_vle], ignore_index=True)

# Split datasets by vapor fraction
df_liquid = combinedDataFrame[combinedDataFrame.q == 0.0].reset_index(drop=True)
df_vapor = combinedDataFrame[combinedDataFrame.q == 1.0].reset_index(drop=True)
df_twophase = combinedDataFrame[
    (combinedDataFrame.q > 0.0) & (combinedDataFrame.q < 1.0)
].reset_index(drop=True)

# Filter only pure liquid and vapor points
# df_liquid_vapor = combinedDataFrame[(combinedDataFrame.q == 0.0) | (combinedDataFrame.q == 1.0)].reset_index(drop=True)

# Save as a single CSV
combinedDataFrame.to_csv(f"{compound}.csv", index=False)

# Save CSVs
# df_liquid.to_csv(f"Saved_{compound}_unscaled_Liquid.csv", index=False)
# df_vapor.to_csv(f"Saved_{compound}_unscaled_Vapor.csv", index=False)
# df_twophase.to_csv(f"Saved_{compound}_unscaled_TwoPhase.csv", index=False)
# combinedDataFrame.to_csv(f"Saved_{compound}_unscaled_Data_Combined.csv", index=False)

print(f"Saved liquid, vapor, and two-phase datasets for {compound}")
