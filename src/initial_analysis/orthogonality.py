import pandas as pd
import numpy as np
import os

df = pd.read_csv(path,sep=",")
df1 = df[2:27]

res = []

for i in range(2, 28):
    res.append(np.dot(df1[i], df1))
product = np.dot(df1,df1.T)
# print("Vector1: {0} , Vector2:{1}".format(vector,vector1))
print(product)
