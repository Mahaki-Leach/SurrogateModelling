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
                    # Not quite correct but works for our purposes
                    PR.update(CP.QT_INPUTS, 0.0, T)
                    lower_p = PR.p()

                    PR.update(CP.QT_INPUTS, 1.0, T)
                    upper_p = PR.p()

                    # sanity & physical order: bubble < dew
                    if any([
                        np.isnan(lower_p), np.isnan(upper_p),
                        lower_p <= 0, upper_p <= 0,
                        lower_p <= upper_p
                    ]):
                        continue

                    # Sweep pressures inside two-phase region
                    for P in np.linspace(lower_p, upper_p, 40):

                        PR.update(CP.PT_INPUTS, P, T)
                        q = PR.Q()

                        # must be in two-phase region
                        if not (0.0 < q < 1.0):
                            continue

                        x_liq = PR.mole_fractions_liquid()
                        x_vap = PR.mole_fractions_vapor()

                        # skip invalid states
                        if any([np.isnan(x_liq).any(), np.isnan(x_vap).any()]):
                            continue

                        # must be valid mole fractions
                        if not (0 <= x_liq[0] <= 1 and 0 <= x_vap[0] <= 1):
                            continue

                        # benzene should be more volatile → y1 > x1
                        if not (x_liq[0] < x_vap[0]):
                            continue

                        # avoid critical-region
                        if abs(x_liq[0] - x_vap[0]) < 1e-2:
                            print("Skipping near-critical point")
                            continue

                        add_row(T, P, x_liq, x_vap, q)

                except Exception:
                    continue

        # generate grid of T
        T_values = np.linspace(Tmin + 2, Tmax - 2, 70)
        VLE_sweep(T_values)

    df = pd.DataFrame(rows).drop_duplicates()
    df.to_csv("interaction_data_refined_clean.csv", index=False)
    return df


df = gen_data()
print("Generated rows:", len(df))