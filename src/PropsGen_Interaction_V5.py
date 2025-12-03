from CoolProp.CoolProp import PropsSI
import CoolProp.CoolProp as CP
import numpy as np
import pandas as pd


def gen_data():

    compound1 = "Benzene"
    compound2 = "Toluene"

    # Query bounds from CoolProp
    Tmin = min(PropsSI("Tmin", compound1), PropsSI("Tmin", compound2))
    Tmax = min(PropsSI("Tcrit", compound1), PropsSI("Tcrit", compound2))

    rows = []

    PR = CP.AbstractState("HEOS", f"{compound1}&{compound2}")

    # Looping through compositions
    for c in np.arange(0.01, 1, 0.01):

        z_c1 = c  # Benzene
        z_c2 = 1 - c # Toluene

        PR.set_mole_fractions([z_c1, z_c2])

        def add_row(T, P, x_liq, x_vap):

            row = {
                "temperature": T,
                "pressure": P,
                "c1_x": z_c1,
                "c1_liq": x_liq[0],
                "c1_vap": x_vap[0]
            }

            rows.append(row)

        def VLE_sweep(T_min, T_max, n_points):

            # Generating even distribution of points
            T_values = np.linspace(T_min, T_max, n_points)
            # P_values = np.linspace(P_min, P_max, n_points)

            for T in T_values:
                try:

                    # Bubble/dew pressures
                    PR.update(CP.QT_INPUTS, 0.0, T) # P < liquid
                    bubble_P = PR.p()

                    PR.update(CP.QT_INPUTS, 1.0, T) # P > vapor
                    dew_P = PR.p()

                    # sanity checks
                    if np.isnan(bubble_P) or np.isnan(dew_P):
                        continue
                    if dew_P <= 0 or bubble_P <= 0:
                        continue
                    if dew_P > bubble_P:
                        continue

                    # Pressures inside two-phase region
                    for P in np.linspace(dew_P, bubble_P, 50):

                        PR.update(CP.PT_INPUTS, P, T)

                        # Append to DataFrame
                        x_liq = PR.mole_fractions_liquid()
                        x_vap = PR.mole_fractions_vapor()

                        # skip invalid states
                        if (np.isnan(x_liq).any() or np.isnan(x_vap).any()):
                            continue
                        if (x_liq[0] < 0) or (x_liq[0] > 1):
                            continue
                        if (x_vap[0] < 0) or (x_vap[0] > 1):
                            continue

                        add_row(T, P, x_liq, x_vap)

                except Exception as e:
                    print(e)
                    continue

        VLE_sweep(Tmin, Tmax, 100)

    df = pd.DataFrame(rows)
    df = df.drop_duplicates()
    df.to_csv("interaction_data.csv", index=False)

gen_data()
