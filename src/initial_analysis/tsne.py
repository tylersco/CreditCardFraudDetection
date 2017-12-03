from datetime import time
from formatter import NullFormatter

import pandas as pd
import sys
from sklearn import manifold
import matplotlib as plt


import numpy as np
import matplotlib.pyplot as plt

from matplotlib.ticker import NullFormatter
from sklearn import manifold, datasets
from time import time

#do one with reduced data
#do one with all data


def main():

    df = pd.read_csv(sys.argv[1]).head(50000)

    # df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1) \
    #     .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1).head(50000)

    t0 = time()

    X_tsne = manifold.TSNE(learning_rate=200, n_iter=1500, verbose=4).fit_transform(df)

    # tsne = manifold.TSNE(n_components=2, init='random', random_state=0)
    plt.figure()
    plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=df['Class'])
    plt.xlabel("x-tsne-pca")
    plt.ylabel("y-tsne-pca")

    t1 = time()

    plt.savefig('tsne_300kpoints_1500iterations_partialfeature.png', bbox_inches='tight')

    print(t1-t0)
    plt.show()

if __name__ == '__main__':
    main()