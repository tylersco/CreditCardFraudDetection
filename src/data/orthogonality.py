#!/usr/bin/python
import pandas as pd
import numpy as np
import os

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path += '/data/creditcard.csv'

df = pd.read_csv(path,sep=",")
df1 = df[1:28]
product = np.dot(df1,df1.T)
# print("Vector1: {0} , Vector2:{1}".format(vector,vector1))
print product

