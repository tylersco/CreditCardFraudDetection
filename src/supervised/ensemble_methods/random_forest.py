import sys
sys.path.append('../')
import os
import time
import pandas as pd
import numpy as np
from sklearn import metrics, model_selection, ensemble
import matplotlib.pyplot as plt
from classifier import Classifier

class RandomForest(Classifier):
    def random_forest(self, X, y, valid, test):
        '''
        n_jobs -1 uses all available cores
        '''
        class_weights = {0: 1, 1: 8}
        rf = ensemble.RandomForestClassifier(n_estimators=30, criterion='gini', min_samples_split=12, n_jobs=-1, class_weight=class_weights)
        start = time.time()
        rf.fit(X, y)
        end = time.time()

        # TRAIN DATA

        # y_score = rf.predict_proba(X)[:, 1]
        # results = rf.predict(X)
        #
        # # Get metrics
        # mets = self.compute_metrics(y, results, y_score)
        #
        # print('AUROC:', mets['auroc'])
        # print('Accuracy:', mets['accuracy'])
        # print('Precision:', mets['precision'])
        # print('Recall:', mets['recall'])
        # print('F Score:', mets['f'])
        # print('Average Precision', mets['ap'])
        # print(mets['confusion'])

        # VALID DATA

        # y_score = rf.predict_proba(valid.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        # results = rf.predict(valid.drop("Class", axis=1).drop("Time", axis=1))
        #
        # # Get metrics
        # mets = self.compute_metrics(valid["Class"], results, y_score)
        #
        # print('AUROC:', mets['auroc'])
        # print('Accuracy:', mets['accuracy'])
        # print('Precision:', mets['precision'])
        # print('Recall:', mets['recall'])
        # print('F Score:', mets['f'])
        # print('Average Precision', mets['ap'])
        # print(mets['confusion'])

        # TEST DATA

        y_score = rf.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        results = rf.predict(test.drop("Class", axis=1).drop("Time", axis=1))

        # Get metrics
        mets = self.compute_metrics(test["Class"], results, y_score)
        mets['time'] = end - start

        print('AUROC:', mets['auroc'])
        print('Accuracy:', mets['accuracy'])
        print('Precision:', mets['precision'])
        print('Recall:', mets['recall'])
        print('F Score:', mets['f'])
        print('Average Precision', mets['ap'])
        print(mets['confusion'], '\n')

        # Precision recall measure
        #self.plot_precision_recall(test["Class"], y_score, 'Random Forest')

        # Plot ROC
        #self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Random Forest')

        return mets


def main():
    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1) \
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    results = {
        'accuracy': [],
        'precision': [],
        'recall': [],
        'f': [],
        'ap': [],
        'auroc': [],
        'confusion': [],
        'fpr': [],
        'tpr': [],
        'time': []
    }

    filepath = os.path.join('..', '..', 'results', 'random_forest_results.txt')
    replications = 20
    for i in range(replications):
        # Create train and test groups
        train, test = model_selection.train_test_split(df, test_size=0.2)
        train, valid = model_selection.train_test_split(train, test_size=0.25)

        # X and Y used for sklearn logreg
        X = train.drop("Class", axis=1).drop("Time", axis=1)
        y = train["Class"]

        r_forest = RandomForest()
        metrics = r_forest.random_forest(X, y, valid, test)

        results['accuracy'].append(metrics['accuracy'])
        results['precision'].append(metrics['precision'])
        results['recall'].append(metrics['recall'])
        results['f'].append(metrics['f'])
        results['ap'].append(metrics['ap'])
        results['auroc'].append(metrics['auroc'])
        results['confusion'].append(metrics['confusion'])
        results['fpr'].append(metrics['fpr'])
        results['tpr'].append(metrics['tpr'])
        results['time'].append(metrics['time'])

    with open(filepath, 'w') as f:
        f.write('Accuracy: ' + str(results['accuracy']) + ': ' + str(np.mean(results['accuracy'])) + ': ' + str(np.std(results['accuracy'])) + '\n')
        f.write('Precision: ' + str(results['precision']) + ': ' + str(np.mean(results['precision'])) + ': ' + str(np.std(results['precision'])) + '\n')
        f.write('Recall: ' + str(results['recall']) + ': ' + str(np.mean(results['recall'])) + ': ' + str(np.std(results['recall'])) + '\n')
        f.write('F-score: ' + str(results['f']) + ': ' + str(np.mean(results['f'])) + ': ' + str(np.std(results['f'])) + '\n')
        f.write('AP: ' + str(results['ap']) + ': ' + str(np.mean(results['ap'])) + ': ' + str(np.std(results['ap'])) + '\n')
        f.write('AUROC: ' + str(results['auroc']) + ': ' + str(np.mean(results['auroc'])) + ': ' + str(np.std(results['auroc'])) + '\n')
        f.write('Time (sec): ' + str(results['time']) + ': ' + str(np.mean(results['time'])) + ': ' + str(np.std(results['time'])) + '\n')

if __name__ == '__main__':
    main()
