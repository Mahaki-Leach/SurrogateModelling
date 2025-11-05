# Import statements
import os
import numpy as np
import pandas as pd

# Import IDAES libraries
from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.pysmo_surrogate import PysmoRBFTrainer, PysmoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_parity,
    surrogate_residual,
)

# Import training data
np.set_printoptions(precision=6, suppress=True)

# Load pure Benzene dataset CSV
csv_data = pd.read_csv("Saved_Benzene_unscaled_Data_Combined.csv")

# Select subset for training (optional)
data = csv_data.sample(n=1000, random_state=42)

# Define input and output columns
input_labels = ["temperature", "pressure", "q"]
output_labels = ["enth_mol", "entr_mol", "vol_mol"]

# Split training and validation data (80/20)
data_training, data_validation = split_training_validation(data, training_fraction=0.8, seed=42)

# Create PySMO RBF trainer object
trainer = PysmoRBFTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# Configure RBF options
trainer.config.basis_function = "gaussian"
trainer.config.solution_method = "algebraic"
trainer.config.regularization = True

# Train the RBF surrogate
rbf_train = trainer.train_surrogate()

# Input bounds from dataset
Tmin, Tmax = data["temperature"].min(), data["temperature"].max()
Pmin, Pmax = data["pressure"].min(), data["pressure"].max()
Qmin, Qmax = 0.0, 1.0
input_bounds = {"temperature": (Tmin, Tmax), "pressure": (Pmin, Pmax), "q": (Qmin, Qmax)}

# Create callable surrogate object
rbf_surr = PysmoSurrogate(rbf_train, input_labels, output_labels, input_bounds)

# Save surrogate to JSON
rbf_surr.save_to_file("pysmo_benzene_VLE.json", overwrite=True)

# Visualize results
surrogate_scatter2D(rbf_surr, data_training, filename="pysmo_benzene_VLE_scatter2D.pdf")
surrogate_parity(rbf_surr, data_training, filename="pysmo_benzene_VLE_parity.pdf")
surrogate_residual(rbf_surr, data_training, filename="pysmo_benzene_VLE_residual.pdf")
