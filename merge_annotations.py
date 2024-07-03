import pandas as pd
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--src', help='Path to the source CSV file with annotations')
parser.add_argument('--dst', help='Path to the destiny CSV file with annotations')
args = parser.parse_args()

# Read the CSV files
df_src = pd.read_csv(args.src)
df_dst = pd.read_csv(args.dst)

# Merge the DataFrames
df_merged = pd.concat([df_src, df_dst])

# Remove duplicates
df_merged.drop_duplicates(inplace=True)

# Save the merged DataFrame to the same file as the destination CSV
df_merged.to_csv(args.dst, index=False)