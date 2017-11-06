import sys

import os

import matplotlib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def main():

    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path += '/data/creditcard.csv'
    df = pd.read_csv(path)

    columns = ['Class', 'Amount']
    df1 = pd.DataFrame(df, columns=columns)



    genuine = df1['Amount'][df['Class'] == 0]

    max_genuine = genuine.max()
    min_genuine = genuine.min()
    mean_genuine = genuine.mean()
    variance_genuine = genuine.var()

    data = np.concatenate((variance_genuine, mean_genuine, max_genuine, min_genuine), 0)

    print(data)

    plt.boxplot(data)
    plt.title("Genuine Transactions")
    plt.show()


if __name__ == '__main__':
    main()

