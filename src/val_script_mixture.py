import numpy as np
import pandas as pd
from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from pyomo.environ import SolverFactory
from PP_mixture_V2 import SurrogateMixtureParameterBlock

from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP


# ---------------------
# Setup
# ---------------------

compound1 = "Benzene"
compound2 = "Toluene"

PR = CP.AbstractState("HEOS", f"{compound1}&{compound2}")

solver = SolverFactory("ipopt")

# ---------------------
# Temperature sweep
# ---------------------

T_vals = np.linspace(300, 582, 20)

records = []

for z in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:

    z1 = z
    z2 = 1 - z
    PR.set_mole_fractions([z1, z2])

    for P in np.linspace(5e4, 1e6, 20):

        for T in T_vals:

            # Skip one-phase region (Q<0 or Q>1)
            try:
                PR.update(CP.PT_INPUTS, P, T)
                Q_ref = PR.Q()
            except:
                continue

            if Q_ref <= 0 or Q_ref >= 1:
                continue

            # Reference values
            h_ref = PR.hmolar()
            s_ref = PR.smolar()

            # ---------------------
            # Surrogate prediction
            # ---------------------

            m = ConcreteModel()
            m.params = SurrogateMixtureParameterBlock()
            m.props = m.params.build_state_block([1], defined_state=True)

            m.props[1].flow_mol.fix(1)
            m.props[1].temperature.fix(T)
            m.props[1].pressure.fix(P)

            m.props[1].mole_frac_comp["benzene"].fix(z1)
            m.props[1].mole_frac_comp["toluene"].fix(z2)

            m.props.initialize()

            Q_surr = value(m.props[1].q)
            h_surr = value(m.props[1].enth_mol)
            s_surr = value(m.props[1].entr_mol)

            records.append(
                {
                    "T": T,
                    "Q_ref": Q_ref,
                    "Q_surr": Q_surr,
                    "h_ref": h_ref,
                    "h_surr": h_surr,
                    "s_ref": s_ref,
                    "s_surr": s_surr,
                }
            )

# ---------------------
# Compute mean relative error
# ---------------------

df = pd.DataFrame(records)

REL_Q = np.mean((df["Q_ref"] - df["Q_surr"]))
REL_h = np.mean((df["h_ref"] - df["h_surr"]))
REL_s = np.mean((df["s_ref"] - df["s_surr"]))

print("\n--- mean REL error RESULTS ---")
print(f"Q REL   = {REL_Q:.6e}")
print(f"h REL   = {REL_h:.6e}")
print(f"s REL   = {REL_s:.6e}")

print(f"\nValidation Complete.{len(df)} points evaluated.")
