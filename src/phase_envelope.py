from CoolProp.CoolProp import AbstractState
import matplotlib.pyplot as plt
import numpy as np

compound1 = "Benzene"
compound2 = "Toluene"

# Create mixture
AS = AbstractState("HEOS", f"{compound1}&{compound2}")
AS.set_mole_fractions([0.5, 0.5])

# Build phase envelope
AS.build_phase_envelope("TP")
PE = AS.get_phase_envelope_data()

# Phase envelope points
T_points = np.array(PE.T)
P_points = np.array(PE.p)

# Plot T-P phase envelope
plt.figure(figsize=(7,5))
plt.plot(T_points, P_points, color="blue")
plt.xlabel("Temperature (K)")
plt.ylabel("Pressure (Pa)")
plt.title("Benzene–Toluene Phase Envelope (CoolProp)")
plt.grid(True)
plt.tight_layout()
plt.show()