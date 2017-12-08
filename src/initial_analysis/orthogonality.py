import pandas as pd
import numpy as np
import os
import sys

df = pd.read_csv(sys.argv[1], sep=",")
df1 = df.iloc[:,1:29]

res = []

# Ensure that dot product between pairwise features is orthogonal
for i in range(0, len(df1.iloc[0,:])):
    r = []
    for j in range(i + 1, len(df1.iloc[0,:])):
        r.append(df1.iloc[:,i].dot(df1.iloc[:,j]))

    if r:
        res.append(r)

# These values are all really close to 0, which means that the columns are all orthogonal to eachother
print(res[0])
