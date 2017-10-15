import pandas as pd
import numpy as np

def load_data(path):
    """
    Load the credit card fraud data

    path: Absolute path to the creditcard.csv file
    returns: Four numpy arrays corresponding to the features and labels for genuine
             transactions and features and labels for fraudulent transactions
    """

    df = pd.read_csv(path, sep=',')

    genuine = df[df['Class'] == 0].values
    fraudulent = df[df['Class'] == 1].values

    x_genuine = genuine[:, :-1]
    y_genuine = np.reshape(genuine[:, -1], (-1, 1))
    x_fraudulent = fraudulent[:, :-1]
    y_fraudulent = np.reshape(fraudulent[:, -1], (-1, 1))

    return x_genuine, y_genuine, x_fraudulent, y_fraudulent
