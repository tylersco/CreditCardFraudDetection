"""
Distribution plots for each feature in the dataset

References:
https://seaborn.pydata.org/tutorial/distributions.html
https://matplotlib.org/gallery/lines_bars_and_markers/markevery_demo.html#sphx-glr-gallery-lines-bars-and-markers-markevery-demo-py
https://matplotlib.org/users/gridspec.html
https://www.kaggle.com/currie32/predicting-fraud-with-tensorflow
"""

import pandas as pd
import sys
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as mpatches
import seaborn

df = pd.read_csv(sys.argv[1], sep=",")
# Grab features
df1 = df.iloc[:,1:29]

# Grid size of 7 rows by 4 columns
gridspec = GridSpec(7, 4)
gridspec.update(hspace=0.8)
for i in range(0, len(df1.iloc[0, :])):
    ax = plt.subplot(gridspec[i])
    # Create distribution plots
    seaborn.distplot(df1[df.Class == 1].iloc[:,i])
    seaborn.distplot(df1[df.Class == 0].iloc[:,i])
    ax.set_title('Feature Distribution by Class: V{0}'.format(i + 1), fontsize=10)
    ax.set_xlabel('', fontsize=10)

# Create legend
blue_patch = mpatches.Patch(color='blue', label='Fraudulent Transactions')
orange_patch = mpatches.Patch(color='orange', label='Genuine Transactions')
plt.figlegend([blue_patch, orange_patch], ['Fraudulent Transactions', 'Genuine Transactions'], loc='upper left')
plt.show()
