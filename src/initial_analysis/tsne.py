from datetime import time
from formatter import NullFormatter

import pandas as pd
import sys
from sklearn import manifold
import matplotlib as plt
import matplotlib.patches as mpatches


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

def main():

    df = pd.read_csv(sys.argv[1])
    df_fraud = df.loc[df['Class'] == 1]
    # Only sample 50,000 data points
    df_gen = df.loc[df['Class'] == 0].head(50000)
    df_new = pd.concat([df_fraud, df_gen])
    # Drop uninformative features
    df_new = df_new.drop('Time', axis=1).drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1).drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    t0 = time()

    # Execute t-sne
    X_tsne = manifold.TSNE(learning_rate=500, n_iter=600, verbose=4).fit_transform(df_new.drop('Class', axis=1))

    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df_new['Class'])
    plt.xlabel("x-tsne")
    plt.ylabel("y-tsne")
    plt.title('T-SNE on Reduced Credit Card Fraud Detection Dataset')

    purple_patch = mpatches.Patch(color='purple', label='Genuine')
    yellow_patch = mpatches.Patch(color='yellow', label='Fraudulent')
    plt.legend(handles=[purple_patch, yellow_patch])

    t1 = time()

    plt.savefig('tsne_50kpoints_600iterations_partialfeature.png', bbox_inches='tight')

    print(t1-t0)
    plt.show()

if __name__ == '__main__':
    main()
