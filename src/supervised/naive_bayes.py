import sys

import os
import pandas as pd
import numpy as np
sys.path.append('../data/')
from sklearn.naive_bayes import GaussianNB as GNB
from sklearn.metrics import confusion_matrix as conf
from sklearn import metrics
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


def naive_bayes(x_train,x_test,y_train,y_test):

    test = GNB();
    test.fit(x_train,y_train)

    result = test.predict(x_test)
    accuracy = test.score(x_test, y_test)
    y_score = test.predict_proba(x_test)[:,1]

    # AUC and ROC measures
    fpr, tpr, thresholds = metrics.roc_curve(y_test, result)
    auc = metrics.auc(fpr, tpr)
    precision = metrics.precision_score(y_test, result)
    recall = metrics.recall_score(y_test, result)
    f_score = metrics.f1_score(y_test, result)

    print('AUROC:', auc)
    print('Accuracy:', accuracy)
    print('Precision:', precision)
    print('Recall:', recall)
    print('F Score:', f_score)

    average_precision = metrics.average_precision_score(y_test, y_score)

    print('Average precision-recall score: {0:0.2f}'.format(
        average_precision))

    precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Naive Bayes 2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    # Plot ROC
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Naive Bayes ROC')
    plt.legend(loc="lower right")
    plt.show()

    #return results, accuracy, confusion

#path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
#path += '/data/creditcard.csv'

# Read in data through command line argument
df = pd.read_csv(sys.argv[1])
df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1)\
    .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)
X = df.drop("Class",axis=1).drop("Time",axis=1)
Y = df["Class"]
df = df.values;

x_train, x_test, y_train, y_test = train_test_split(X,Y,test_size=0.2)

naive_bayes(x_train,x_test,y_train,y_test)
