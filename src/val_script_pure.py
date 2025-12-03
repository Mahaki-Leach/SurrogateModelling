import numpy as np
import pandas as pd
from pyomo.environ import ConcreteModel, value
from PP_v2 import SurrogateParameterBlock  # adjust import path
from idaes.core.surrogate.pysmo_surrogate import PysmoSurrogate
from CoolProp.CoolProp import PropsSI, PhaseSI

# ---------------------
# Setup
# ---------------------

compound = "Benzene"
mw = 0.07811  # kg/mol

# Load trained surrogate
surr = PysmoSurrogate.load_from_file("Benzene.json")

# Temperature and pressure ranges
T_vals = np.linspace(300, 550, 20)   # K
P_vals = np.linspace(101325, 101325*10, 5)  # Pa

records = []

# ---------------------
# Loop over T and P
# ---------------------

for T in T_vals:
    for P in P_vals:

        # Reference CoolProp values
        try:
            h_ref = PropsSI("Hmolar", "T", T, "P", P, compound)
            s_ref = PropsSI("Smolar", "T", T, "P", P, compound)
            d_ref = PropsSI("Dmolar", "T", T, "P", P, compound)
            q_ref = PropsSI("Q", "T", T, "P", P, compound)

            # Handle one-phase states
            if q_ref == -1:
                phase = PhaseSI("T", T, "P", P, compound)
                if "liquid" in phase.lower():
                    q_ref = 0.0
                elif "gas" in phase.lower() or "vapor" in phase.lower():
                    q_ref = 1.0
                else:
                    q_ref = np.nan

        except Exception as e:
            print(f"CoolProp failed at T={T}, P={P}: {e}")
            continue

        # ---------------------
        # Surrogate prediction
        # ---------------------
        try:
            m = ConcreteModel()
            m.params = SurrogateParameterBlock(surrogate=surr, mw=mw)
            m.props = m.params.build_state_block([1], defined_state=True)

            m.props[1].flow_mol.fix(1)
            m.props[1].temperature.fix(T)
            m.props[1].pressure.fix(P)
            m.props[1].q.fix(q_ref)

            m.props.initialize(outlvl=0)

            h_surr = value(m.props[1].enth_mol)
            s_surr = value(m.props[1].entr_mol)
            d_surr = 1 / value(m.props[1].vol_mol)

            records.append({
                "T": T, "P": P,
                "h_ref": h_ref, "h_surr": h_surr,
                "s_ref": s_ref, "s_surr": s_surr,
                "d_ref": d_ref, "d_surr": d_surr
            })

        except Exception as e:
            print(f"Surrogate failed at T={T}, P={P}: {e}")
            continue

# ---------------------
# Convert to DataFrame
# ---------------------
df = pd.DataFrame(records)

# Compute RMSE
rmse_h = np.sqrt(np.mean((df["h_ref"] - df["h_surr"])**2))
rmse_s = np.sqrt(np.mean((df["s_ref"] - df["s_surr"])**2))
rmse_d = np.sqrt(np.mean((df["d_ref"] - df["d_surr"])**2))

print("\n--- Pure Surrogate Validation ---")
print(f"h RMSE   = {rmse_h:.6e} J/mol")
print(f"s RMSE   = {rmse_s:.6e} J/mol-K")
print(f"d RMSE   = {rmse_d:.6e} mol/m^3")