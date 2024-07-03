import pandas as pd
import os
import shutil
import argparse

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--src', help='Path to the CSV file with annotations')
parser.add_argument('--dst', help='Path to the new CSV file')
args = parser.parse_args()

classes = ['CCAP', 'AGL', 'AGL', 'ALC', 'AST', 'BAS', 'BEC', 'BOVO', 'CANC', 'DIP', 'DROE', 'DSN', 'DVE', 'ELS', 'HENR', 'ISC', 'ISS', 'MED', 'MSC', 'Rock', 'SHAD', 'URF']
# classes = ['CCAP']

# Read the CSV file
df = pd.read_csv(args.src)

# Remove leading and trailing spaces from 'Label' values
df['Label'] = df['Label'].str.strip()

# Filter the DataFrame
df_filtered = df[df['Label'].isin(classes)]

# Save the filtered DataFrame to a new CSV file
df_filtered.to_csv(args.dst, index=False)