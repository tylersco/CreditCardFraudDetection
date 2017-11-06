import sys

import os
import pandas as pd
import numpy as np
sys.path.append('../data/')
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import confusion_matrix as conf


def naive_bayes(X,Y):

    test = GNB();
    test.fit(X,Y)
    result = test.predict(X)
    print result
    print test.score(X,Y)
    print conf(Y,result)

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path += '/data/creditcard.csv'
df = pd.read_csv(path)
X = df.drop("Class",axis=1).drop("Time",axis=1)
Y = df["Class"]
df = df.values;

naive_bayes(X,Y)

