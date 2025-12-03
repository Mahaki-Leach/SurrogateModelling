from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd


def gen_data():

    compound1 = "Benzene"
    compound2 = "Toluene"

    # Query bounds
    Tmin = max(PropsSI("Tmin", compound1), PropsSI("Tmin", compound2))
    Tmax = min(PropsSI("Tcrit", compound1), PropsSI("Tcrit", compound2))

    rows = []
    PR = CP.AbstractState("HEOS", f"{compound1}&{compound2}")

    # Loop compositions
    for c in np.linspace(0.05, 0.95, 19):

        z_c1 = c
        z_c2 = 1 - c
        PR.set_mole_fractions([z_c1, z_c2])

        def add_row(T, P, x_liq, x_vap, q):
            rows.append({
                "temperature": T,
                "pressure": P,
                "c1_x": z_c1,
                "c1_liq": x_liq[0],
                "c1_vap": x_vap[0],
                "Q": q
            })

        def VLE_sweep(T_values):

            for T in T_values:
                try:
                    for Q in np.linspace(0.01, 0.99, 100):
                        PR.update(CP.QT_INPUTS, Q, T)

                        P = PR.p()

                        x_liq = PR.mole_fractions_liquid()
                        x_vap = PR.mole_fractions_vapor()

                        # skip invalid states
                        if any([np.isnan(x_liq).any(), np.isnan(x_vap).any()]):
                            print("Invalid state")
                            continue

                        # must be valid mole fractions
                        if not (0 <= x_liq[0] <= 1 and 0 <= x_vap[0] <= 1):
                            print("Invalid mole fractions")
                            continue

                        # benzene should be more volatile → y1 > x1
                        if not (x_liq[0] < x_vap[0]):
                            print("Skipping non-volatile state")
                            continue

                        # avoid critical-region
                        if abs(x_liq[0] - x_vap[0]) < 1e-2:
                            print("Skipping near-critical point")
                            continue

                        add_row(T, P, x_liq, x_vap, Q)

                except Exception:
                    continue

        # generate grid of T
        VLE_sweep(np.linspace(Tmin, Tmax, 100))

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv("interaction_data_q_based.csv", index=False)
    return df


df = gen_data()
print("Generated rows:", len(df))