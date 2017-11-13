import pandas as pd
from pandas import *
import numpy as np
from scipy.stats.stats import pearsonr
import itertools
# import matplotlib.pyplot as plt
import sys


def correlation(df, csv_name):
  correlation_df = df.corr(method='pearson')
  print(correlation_df.to_string())
  correlation_df.to_csv(csv_name)
  return correlation_df

def fraud_correlation(df):
  df_fraud = pd.DataFrame()
  df_fraud = pd.concat([df_fraud, df[df['Class'] == 1]])
  #print(df_fraud.to_string())
  correlation_df_fraud = correlation(df_fraud, "ccf_fraud_correlation.csv")

def nonfraud_correlation(df):
  df_nonfraud = pd.DataFrame()
  df_nonfraud = pd.concat([df_nonfraud, df[df['Class'] == 0]])
  #print(df_nonfraud.to_string())
  correlation_df_nonfraud = correlation(df_nonfraud, "ccf_nonfraud_correlation.csv")

if __name__ == '__main__':
  filename = sys.argv[1]
  df = pd.read_csv(filename)
  #correlation(df)
  fraud_correlation(df)
  nonfraud_correlation(df)