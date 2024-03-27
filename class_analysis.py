import pandas as pd
import numpy as np
import sys
import os

# Read in the data from the csv file from command line
df = pd.read_csv(sys.argv[1])

# Read the image directory
image_dir = sys.argv[2]

# Get the images that are both in the csv and in the directory
df = df[df['Name'].isin([f.replace('.jpg', '') for f in os.listdir(image_dir) if f.endswith('.jpg')])]

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)

# Number of unique images
print("Number of images:",len(df['Name'].unique()))

# Number of unique labels
print("Number of labels:",len(df['Label'].unique()))

# Number of total annotations
print("Number of annotations:",len(df))

counts = df['Label'].value_counts()
percentages = df['Label'].value_counts(normalize=True).mul(100).round(1).astype(str) + '%'
percentages_df = pd.DataFrame({'counts': counts, 'percentage': percentages})
print(percentages_df)

# Create dataframe that contains the Images x Labels matrix
df_matrix = df.groupby(['Name','Label']).size().unstack(fill_value=0)
df_matrix = df_matrix.reset_index()
df_matrix = df_matrix.set_index('Name')

# Save the matrix to a csv file
df_matrix.to_csv('Sebens_MA_LTM.csv')