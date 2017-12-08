import sys
import os
import time
import pandas as pd
import numpy as np
from sklearn import svm, metrics, model_selection
import matplotlib.pyplot as plt
from classifier import Classifier

# Support vector machine classifier
class support_Vector_Machine(Classifier):
    def svm(self, X, y, valid, test):

        # Weights associated with each class
        class_weights = {0: 1, 1: 8}
        # Run SVM classifier with RBF kernel
        svm_model = svm.SVC(kernel='rbf', class_weight=class_weights, C=1.0, verbose=3, max_iter=1000, probability=True)

        start = time.time()
        svm_model.fit(X, y)
        end = time.time()

        # TRAIN DATA

        # y_score = svm_model.predict_proba(X)[:, 1]
        # results = svm_model.predict(X)
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
        # print(mets['confusion'], '\n')

        #VALID DATA

        # y_score = svm_model.predict_proba(valid.drop("Class", axis=1))[:, 1]
        # results = svm_model.predict(valid.drop("Class", axis=1))
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
        # print(mets['confusion'], '\n')
        # return mets

        # TEST DATA

        y_score = svm_model.predict_proba(test.drop("Class", axis=1))[:, 1]
        results = svm_model.predict(test.drop("Class", axis=1))

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
        #
        # # Precision recall measure
        # #self.plot_precision_recall(test["Class"], y_score, 'Logistic Regression')
        #
        # # Plot ROC
        # #self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Logistic Regression')
        #
        return mets


def main():

    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1)\
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1).drop('Time', axis=1)

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

    filepath = os.path.join('..', 'results', 'svm.txt')
    replications = 20
    for i in range(replications):
        # Create train and test groups
        train, test = model_selection.train_test_split(df, test_size=0.2)
        train, valid = model_selection.train_test_split(train, test_size=0.25)

        X = train.drop("Class", axis=1)
        y = train["Class"]

        SVM = support_Vector_Machine()
        metrics = SVM.svm(X, y, valid, test)

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

    # Write results to file
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