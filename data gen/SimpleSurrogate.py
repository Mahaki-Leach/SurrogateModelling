# Import statements
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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

csv_data = pd.read_csv("Simple_Data.csv")

csv_data.columns.values[0:4] = [
    "a",
    "b",
    "c",
    "d"
]

# Data are floating point numbers

data = csv_data.sample(n=1000)

print(data)

# Getting input and output data
input_data = data.iloc[:, 0:2] # a and b
output_data = data.iloc[:, 2:4] # c and d

# Define labels, and split training and validation data
input_labels = list(input_data.columns)
output_labels = list(output_data.columns)

print("Input labels:", input_labels)
print("Output labels:", output_labels)

n_data = data[input_labels[0]].size

data_training, data_validation = split_training_validation(data, 0.8, seed=n_data)

# # Create PySMO trainer object
trainer = PysmoPolyTrainer(
    input_labels=input_labels,
    output_labels=output_labels,
    training_dataframe=data_training,
)

# # Set PySMO trainer options
trainer.config.maximum_polynomial_order = 5
trainer.config.multinomials = True
trainer.config.training_split = 0.8
trainer.config.number_of_crossvalidations = 10

# # Train surrogate (calls PySMO through IDAES Python wrapper)
poly_train = trainer.train_surrogate()

print(type(poly_train))
print(poly_train.num_outputs)

# # create callable surrogate object
xmin, xmax = [1, 1], [100, 100]
input_bounds = {input_labels[i]: (xmin[i], xmax[i]) for i in range(len(input_labels))}

print(input_bounds)

poly_surr = PysmoSurrogate(poly_train, input_labels, output_labels, input_bounds)

# print(poly_train.output_models.keys())

print(input_data.head())
print(input_data.dtypes)

poly_surr.evaluate_surrogate(input_data)

# surrogate_parity(poly_surr, data_validation, filename="pysmo_poly_val_parity.pdf")
# # # Plotting the surrogate model
# # surrogate_parity(poly_surr, data_validation, filename="parity.pdf", show=True)