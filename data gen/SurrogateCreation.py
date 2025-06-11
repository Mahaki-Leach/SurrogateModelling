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

csv_data = pd.read_csv("Saved_Mixture_Data.csv")

csv_data.columns.values[0:6] = [
    "temperature",
    "pressure",
    "mole_frac_benzene",
    "mole_frac_toluene",
    "enth_mol",
    "entr_mol",
]

data = csv_data.sample(n=1000)

input_data = data.iloc[:, 0:4]
output_data = data.iloc[:, 4:6]

# Define labels, and split training and validation data
input_labels = list(input_data.columns)
output_labels = list(output_data.columns)

n_data = data[input_labels[0]].size

data_training, data_validation = split_training_validation(data, 0.8, seed=n_data)

# Create PySMO trainer object
trainer = PysmoPolyTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# Set PySMO trainer options
trainer.config.maximum_polynomial_order = 3
trainer.config.multinomials = True
trainer.config.training_split = 0.8
trainer.config.number_of_crossvalidations = 10

# Train surrogate (calls PySMO through IDAES Python wrapper)
poly_train = trainer.train_surrogate()

# create callable surrogate object
xmin, xmax = [273.15, 1000, 0.2, 0.2], [273.15+400, 900000, 0.8, 0.8]
input_bounds = {input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))}
poly_surr = PysmoSurrogate(poly_train, input_labels, output_labels, input_bounds)

# save model to JSON
model = poly_surr.save_to_file("pysmo_mixture.json", overwrite=True)

# visualize with IDAES surrogate plotting tools
# surrogate_scatter2D(poly_surr, data_training, filename="pysmo_poly_train_scatter2D.pdf")
# surrogate_parity(poly_surr, data_training, filename="pysmo_poly_train_parity.pdf")
# surrogate_residual(poly_surr, data_training, filename="pysmo_poly_train_residual.pdf")

# visualize with IDAES surrogate plotting tools
# surrogate_scatter2D(poly_surr, data_validation, filename="pysmo_poly_val_scatter2D.pdf")
surrogate_parity(poly_surr, data_validation, filename="pysmo_poly_val_parity.pdf")
# surrogate_residual(poly_surr, data_validation, filename="pysmo_poly_val_residual.pdf")