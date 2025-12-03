from pyomo.environ import ConcreteModel, value
from idaes.core import FlowsheetBlock
from pyomo.environ import SolverFactory
from PP_mixture_V2 import SurrogateMixtureParameterBlock
from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd

solver = SolverFactory("ipopt")

compound1 = "Benzene"
compound2 = "Toluene"

PR = CP.AbstractState("HEOS", f"{compound1}&{compound2}")
z1 = 0.5
z2 = 0.5
PR.set_mole_fractions([z1, z2])

results = []

for T in np.linspace(300, 582, 120):

    # --- CoolProp Reference ---
    try:
        PR.update(CP.PT_INPUTS, 101325, T)
        Q_ref = PR.Q()

        # Skip single-phase states (Q < 0 or Q > 1)
        if Q_ref < 0 or Q_ref > 1:
            continue

        x_liq = PR.mole_fractions_liquid()
        y_vap = PR.mole_fractions_vapor()
        h_ref = PR.hmolar()
        s_ref = PR.smolar()

    except:
        continue

    # --- Surrogate Model ---
    m = ConcreteModel()
    m.params = SurrogateMixtureParameterBlock()
    m.props = m.params.build_state_block([1], defined_state=True)

    m.props[1].flow_mol.fix(1)
    m.props[1].temperature.fix(T)
    m.props[1].pressure.fix(101325)
    m.props[1].mole_frac_comp["benzene"].fix(z1)
    m.props[1].mole_frac_comp["toluene"].fix(z2)

    m.props.initialize()

    Q_sur = value(m.props[1].q)
    h_sur = value(m.props[1].enth_mol)
    s_sur = value(m.props[1].entr_mol)

    # Store
    results.append({
        "T": T,
        "Q_ref": Q_ref,
        "Q_sur": Q_sur,
        "h_ref": h_ref,
        "h_sur": h_sur,
        "s_ref": s_ref,
        "s_sur": s_sur,
        "x1_ref": x_liq[0],
        "y1_ref": y_vap[0],
    })

df = pd.DataFrame(results)
df.to_csv("surrogate_validation.csv", index=False)
print(df)
