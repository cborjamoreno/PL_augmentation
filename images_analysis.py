import pandas as pd
import numpy as np
import sys

# Read in the data from the csv file from command line
df = pd.read_csv(sys.argv[1])
pd.set_option('display.max_rows', None)

# Order images by number of different classes on each image
df = df.groupby('Name').nunique().sort_values(by=['Label'], ascending=False)

# Delete Row Column ans Label columns and rename the index column
df = df.drop(columns=['Row', 'Column'])

df = df.rename(columns={'Name': 'Image'})
df = df.rename(columns={'Label': 'Number of Classes'})

print(df)

