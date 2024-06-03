import pandas as pd

# Step 1: Read each CSV file into a DataFrame
df1 = pd.read_csv('Data_PS_Labeled.csv', header=None)
df2 = pd.read_csv('Data_PET_Labeled.csv', header=None)
df3 = pd.read_csv('Data_LDPE_Labeled.csv', header=None)
df4 = pd.read_csv('Data_PVC_Labeled.csv', header=None)

# Step 2: Concatenate the DataFrames in the order you want
# Assuming you want the data from csv_file_1.csv first, followed by csv_file_2.csv, csv_file_3.csv, and csv_file_4.csv
combined_df = pd.concat([df1, df2, df3, df4], ignore_index=True)

# Step 3: Write the combined DataFrame to a new CSV file
combined_df.to_csv('combined_file.csv', index=False, header=False)
