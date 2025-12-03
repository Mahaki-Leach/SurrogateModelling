# Import statements
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

# Import IDAES libraries
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
)

# Import training data
np.set_printoptions(precision=6, suppress=True)

# Load pure Benzene dataset CSV
csv_data = pd.read_csv("interaction_data_q_based.csv")

# Select subset for training (optional)
data = csv_data.sample(n=2000, random_state=100)

# Define input and output columns
input_labels = ["temperature", "pressure", "c1_x"]
output_labels = ["Q", "c1_liq", "c1_vap"]

# Split training and validation data (80/20)
data_training, data_validation = split_training_validation(data, training_fraction=0.8, seed=42)

# Create PySMO RBF trainer object
trainer = PysmoRBFTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# Configure RBF options
trainer.config.basis_function = "linear"
trainer.config.solution_method = "algebraic"
trainer.config.regularization = True

# Train the RBF surrogate
rbf_train = trainer.train_surrogate()

# Input bounds from dataset
Tmin, Tmax = data["temperature"].min(), data["temperature"].max()
Pmin, Pmax = data["pressure"].min(), data["pressure"].max()
input_bounds = {"temperature": (Tmin, Tmax), "pressure": (Pmin, Pmax), "c1_x": (0.0, 1.0)}

# Create callable surrogate object
rbf_surr = PysmoSurrogate(rbf_train, input_labels, output_labels, input_bounds)

# Save surrogate to JSON
rbf_surr.save_to_file("ignore.json", overwrite=True)

# Visualize results
surrogate_scatter2D(rbf_surr, data_training, filename="ignore.pdf")

# Evaluate surrogate on training data
y_train_pred_df = rbf_surr.evaluate_surrogate(data_training[input_labels])

import matplotlib.pyplot as plt

# Prediction error for c1_liq
plt.scatter(
    data_training['c1_x'],
    y_train_pred_df['c1_liq'] - data_training['c1_liq'],
    s=10,
    alpha=0.6
)
plt.xlabel('c1_x')
plt.ylabel('Prediction error (liq)')
plt.show()

# Prediction error for c1_vap
plt.scatter(
    data_training['c1_x'],
    y_train_pred_df['c1_vap'] - data_training['c1_vap'],
    s=10,
    alpha=0.6,
    color='red'
)
plt.xlabel('c1_x')
plt.ylabel('Prediction error (vap)')
plt.show()

# Prediction error for Q
plt.scatter(
    data_training['c1_x'],
    y_train_pred_df['Q'] - data_training['Q'],
    s=10,
    alpha=0.6,
    color='green'
)
plt.xlabel('Q')
plt.ylabel('Prediction error (Q)')
plt.show()



