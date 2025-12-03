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
    Pmin = min(PropsSI("pmin", compound1), PropsSI("pmin", compound2))
    Pmax = min(PropsSI("pcrit", compound1), PropsSI("pcrit", compound2))

    rows = []

    PR = CP.AbstractState("HEOS", f"{compound1}&{compound2}")

    # Looping through compositions
    for c in np.arange(0.01, 1, 0.01):

        z_c1 = c  # Benzene
        z_c2 = 1 - c # Toluene

        PR.set_mole_fractions([z_c1, z_c2])

        def debug_VLE(T_list, P_list, Q_list):
            import matplotlib.pyplot as plt

            plt.figure(figsize=(7, 6))

            sc = plt.scatter(
                T_list,
                P_list,
                c=Q_list,
                cmap='viridis',
                s=8,
                alpha=0.8,
                edgecolors='none'
            )

            plt.colorbar(sc, label='Quality Q')

            plt.xlabel("Temperature (K)")
            plt.ylabel("Pressure (Pa)")
            plt.title("VLE envelope")

            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()

        def add_row(T, P, Q, x_liq, x_vap):
            row = {
                "temperature": T,
                "pressure": P,
                "c1_x": z_c1,
                "c2_x": z_c2,
                "mixture_q": Q,
                "c1_q_liq": x_liq[0],
                "c2_q_liq": x_liq[1],
                "c1_q_vap": x_vap[0],
                "c2_q_vap": x_vap[1]
            }
            rows.append(row)

        def VLE_sweep(T_min, T_max, P_min, P_max,n_points):

            # Generating even distribution of points
            T_values = np.linspace(T_min, T_max, n_points)

            for T in T_values:
                try:

                    # Bubble/dew pressures
                    PR.update(CP.QT_INPUTS, 0.0, T) # P < liquid
                    bubble_P = PR.p()

                    PR.update(CP.QT_INPUTS, 1.0, T) # P > vapor
                    dew_P = PR.p()

                    # Pressures inside two-phase region
                    for P in np.linspace(dew_P, bubble_P, 50):

                        PR.update(CP.PT_INPUTS, P, T)

                        # Append to DataFrame
                        x_liq = PR.mole_fractions_liquid()
                        x_vap = PR.mole_fractions_vapor()

                        Q = None
                        if PR.Q() < 0:
                            continue # skip invalid points
                        else:
                            Q = PR.Q()

                        add_row(T, P, Q, x_liq, x_vap)

                    # Pressures outside the two-phase region

                    # Vapor
                    for P in np.linspace(P_min, bubble_P, 10):

                        if P == bubble_P:
                            continue  # Skip exact bubble point

                        PR.update(CP.PT_INPUTS, P, T)

                        # Append to DataFrame
                        x_liq = PR.mole_fractions_liquid()
                        x_vap = PR.mole_fractions_vapor()

                        Q = None
                        if PR.Q() < 0:
                            # Expected Behaviour: Single phase
                            Q = 1.0
                        else:
                            Q = PR.Q()

                        add_row(T, P, Q, x_liq, x_vap)

                    # Liquid
                    for P in np.linspace(dew_P, P_max, 10):

                        if P == dew_P:
                            continue  # Skip exact dew point

                        PR.update(CP.PT_INPUTS, P, T)

                        # Append to DataFrame
                        x_liq = PR.mole_fractions_liquid()
                        x_vap = PR.mole_fractions_vapor()

                        Q = None
                        if PR.Q() < 0:
                            # Expected Behaviour: Single phase
                            Q = 0.0
                        else:
                            Q = PR.Q()

                        add_row(T, P, Q, x_liq, x_vap)

                except Exception as e:
                    # Invalid state
                    continue

        VLE_sweep(Tmin, Tmax, Pmin, Pmax, 100)

    df = pd.DataFrame(rows)
    df.to_csv("PropsGen_Interaction_Benzene_Toluene_v2.csv", index=False)

    # debug_VLE(df["temperature"], df["pressure"], df["mixture_q"])


gen_data()
