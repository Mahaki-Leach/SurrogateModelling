# Import statements
import os
import numpy as np
import pandas as pd

# Import IDAES libraries
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.pysmo_surrogate import PysmoPolyTrainer, PysmoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_parity,
    surrogate_residual,
)

# Import training data
np.set_printoptions(precision=6, suppress=True)

# Load pure Benzene dataset CSV
csv_data = pd.read_csv("Saved_Benzenev2_Data_Combined.csv")

# Select subset for training (optional)
data = csv_data.sample(n=10000, random_state=42)

# Define input and output columns
input_labels = ["temperature", "pressure", "q"]  # now including vapor fraction
output_labels = ["z"] # using transformed vol_mol

# Split training and validation data (80/20)
data_training, data_validation = split_training_validation(data, training_fraction=0.8, seed=42)

# Create PySMO trainer object
trainer = PysmoPolyTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# Set trainer options
trainer.config.maximum_polynomial_order = 5
trainer.config.multinomials = True
trainer.config.training_split = 0.8
trainer.config.number_of_crossvalidations = 10

# Scaling data by min-max normalization
for col in input_labels:
    data_training[col] = (data_training[col] - data_training[col].min()) / (data_training[col].max() - data_training[col].min())

for col in output_labels:
    data_training[col] = (data_training[col] - data_training[col].min()) / (data_training[col].max() - data_training[col].min())

# Train surrogate
poly_train = trainer.train_surrogate()

# Define input bounds from dataset
Tmin, Tmax = data["temperature"].min(), data["temperature"].max()
Pmin, Pmax = data["pressure"].min(), data["pressure"].max()
Qmin, Qmax = 0.0, 1.0
input_bounds = {"temperature": (Tmin, Tmax), "pressure": (Pmin, Pmax), "q": (Qmin, Qmax)}

# Create callable surrogate object
poly_surr = PysmoSurrogate(poly_train, input_labels, output_labels, input_bounds)

# Save surrogate to JSON
poly_surr.save_to_file("pysmo_toluene.json", overwrite=True)

# Visualize results
surrogate_scatter2D(poly_surr, data_training, filename="pysmo_toluene_scatter2D.pdf")
surrogate_parity(poly_surr, data_training, filename="pysmo_toluene_parity.pdf")
surrogate_residual(poly_surr, data_training, filename="pysmo_toluene_residual.pdf")

print("Toluene surrogate (inputs: T, P, q) training complete and saved to 'pysmo_toluene.json'")