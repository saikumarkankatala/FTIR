import pandas as pd

# Step 1: Read the CSV file
df = pd.read_csv('Data_PVC_Raw.csv', header=None)

# Step 2: Append a 2 to each row
# Create a DataFrame with a single column of twos with the same number of rows as df
twos = pd.DataFrame([3] * len(df), columns=['Three'])

# Concatenate the original DataFrame with the DataFrame of twos
df = pd.concat([df, twos], axis=1)

# Step 3: Write the modified DataFrame back to a CSV file
df.to_csv('Data_PVC_Labeled.csv', index=False, header=False)