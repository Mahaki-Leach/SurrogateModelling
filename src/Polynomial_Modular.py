from idaes.core.surrogate.sampling.data_utils import split_training_validation
from idaes.core.surrogate.pysmo_surrogate import PysmoPolyTrainer, PysmoSurrogate
from idaes.core.surrogate.plotting.sm_plotter import (
    surrogate_scatter2D,
    surrogate_parity,
    surrogate_residual,
)

import numpy as np
import pandas as pd

compound = "Benzene"

def train_surrogate(csv_file, name, input_cols, output_cols):

    data = pd.read_csv(csv_file)

    # Split training/validation
    data_training, data_validation = split_training_validation(data, training_fraction=0.8, seed=42)

    # Min-max normalize inputs
    for col in input_cols:
        if col == "q":
            continue
        data_training[col] = (data_training[col] - data_training[col].min()) / (data_training[col].max() - data_training[col].min())

    # Min-max normalize outputs
    for col in output_cols:
        if col == "z":
            continue
        data_training[col] = (data_training[col] - data_training[col].min()) / (data_training[col].max() - data_training[col].min())

    # Create trainer
    trainer = PysmoPolyTrainer(
        input_labels=input_cols,
        output_labels=output_cols,
        training_dataframe=data_training,
    )
    trainer.config.maximum_polynomial_order = 5
    trainer.config.multinomials = True
    trainer.config.number_of_crossvalidations = 10

    poly_train = trainer.train_surrogate()

    # Input bounds from dataset
    input_bounds = {col: (data[col].min(), data[col].max()) for col in input_cols}

    # Create callable surrogate
    poly_surr = PysmoSurrogate(poly_train, input_cols, output_cols, input_bounds)

    # Save surrogate to JSON
    poly_surr.save_to_file(f"{name}_v2.json", overwrite=True)

    # Optional plotting
    surrogate_scatter2D(poly_surr, data_training, filename=f"{name}_scatter2D.pdf")
    surrogate_parity(poly_surr, data_training, filename=f"{name}_parity.pdf")
    surrogate_residual(poly_surr, data_training, filename=f"{name}_residual.pdf")

    print(f"{name} surrogate training complete.")

# Single-phase surrogates: use only temperature and pressure as inputs
train_surrogate(f"Saved_{compound}_Liquid.csv", f"{compound}_Liquid", input_cols=["temperature", "pressure"], output_cols=["enth_mol", "entr_mol", "z"])
train_surrogate(f"Saved_{compound}_Vapor.csv", f"{compound}_Vapor", input_cols=["temperature", "pressure"], output_cols=["enth_mol", "entr_mol", "z"])

# Two-phase surrogate: include q as input
train_surrogate(f"Saved_{compound}_TwoPhase.csv", f"{compound}_TwoPhase", input_cols=["temperature", "pressure", "q"], output_cols=["enth_mol", "entr_mol"])
