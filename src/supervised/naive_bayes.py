import sys

import os
import pandas as pd
import numpy as np
sys.path.append('../data/')
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import confusion_matrix as conf
from sklearn.model_selection import train_test_split


def naive_bayes(x_train,x_test,y_train,y_test):

    test = GNB();
    test.fit(x_train,y_train)
    result = test.predict(x_test)
    print result
    print test.score(x_test, y_test)
    print conf(y_test, result)

path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
path += '/data/creditcard.csv'
df = pd.read_csv(path)
X = df.drop("Class",axis=1).drop("Time",axis=1)
Y = df["Class"]
df = df.values;
x_train, x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)

naive_bayes(x_train,x_test,y_train,y_test)


