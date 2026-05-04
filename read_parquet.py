import pandas as pd
import numpy as np

# Read parquet file into a DataFrame
df = pd.read_parquet("./dataset/FEMTO/train_1p.parquet")


print(df.head())
# Get first row's "samples"
first_sample = df.loc[0, "samples"]

print(type(first_sample))   # likely list
print(len(first_sample))    # number of timesteps / rows

# convert to numpy for easier shape handling
arr = np.array(first_sample)
print("Shape of first sample:", arr.shape)

# examine contents
print("First 5 timesteps:\n", arr[:5])

# if it's 2D, look at one row (first timestep)
print("Shape of first timestep:", arr[0].shape)
print("First timestep values:", arr[0])

