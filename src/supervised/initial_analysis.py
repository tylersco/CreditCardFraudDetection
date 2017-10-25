import sys

import os
import pandas as pd
import numpy as np
import itertools
sys.path.append('../data/')

# Create a function that calculates p(x | y):
def p_x_given_y(x, mean_y, variance_y):

    # Input the arguments into a probability density function
    p = 1/(np.sqrt(2*np.pi*variance_y)) * np.exp((-(x-mean_y)**2)/(2*variance_y))

    # return p
    return p

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path += '/data/creditcard.csv'

df = pd.read_csv(path)

# number of genuine transactions
n_genuine = df['Class'][df['Class'] == 0].count()
print(n_genuine)

# number of fraudulent transactions
n_fraudulent = df['Class'][df['Class'] == 1].count()
print(n_fraudulent)

# number of total transactions
n_total = df['Class'].count()
print(n_total)

# Group the data by transaction type and calc the means for each feature
data_means = df.groupby('Class').mean()
print(data_means)

''' From the above found that the average fraudulent transaction is more 
        than the average genuine transaction'''

# Group the data by transaction type and calc the variance of each feature
data_variance = df.groupby('Class').var()
print(data_variance)

frauds_variance = []
genuines_variance = []
frauds_mean = []
genuines_mean = []

for each in data_means:
    genuines_mean.append(data_means[each][data_variance.index == 0].values[0])
    frauds_mean.append(data_means[each][data_variance.index == 1].values[0])

    genuines_variance.append(data_variance[each][data_variance.index == 0].values[0])
    frauds_variance.append(data_variance[each][data_variance.index == 1].values[0])

for index, row in df.iterrows():
    # if index > 10:
    #     break
    # print(type(row))
    #
    # print(row["Class"])


    prob_fraud = 1
    for data, var, mean in zip(row, frauds_variance, frauds_mean):
        prob_fraud *= p_x_given_y(data, mean, var)
    # print(prob_fraud)


    prob_genuine = 1
    for data, var, mean in zip(row, frauds_variance, frauds_mean):
        prob_genuine *= p_x_given_y(data, mean, var)
    # print(prob_genuine)

    if prob_fraud > prob_genuine:
        print("Fraud fam")
    else:
        print("not fraud u good")


