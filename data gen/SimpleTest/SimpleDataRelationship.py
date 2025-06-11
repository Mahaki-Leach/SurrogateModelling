import numpy as np
import pandas as pd
from CoolProp.CoolProp import PropsSI
import numpy.random as rand
import time

# Constants
size = 10000

# Randomized inputs
randDB1 = rand.uniform(1, 100, size).round(1)  # Whole integers
randDB2 = rand.uniform(1, 100, size).round(1)  # Whole integers


#### hidden relationships between variables
#
# a = value
# b = value
# c = value
# d = value
#
# c = 4 * a + b
# d = 3 / a - b
#
#### checking how well it regresses

# Setup DataFrame
exportDataFrame = pd.DataFrame({
    # Inputs
    "a": randDB1,
    "b": randDB2,
    # Outputs
    "c": 4 * randDB1 + randDB2,
    "d": 3 / randDB1 - randDB2,
})

# Save to file
exportDataFrame.to_csv("Simple_Data.csv", index=False)