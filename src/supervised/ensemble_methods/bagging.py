import sys
sys.path.append('../')
import pandas as pd
from sklearn import metrics, model_selection, ensemble
import matplotlib.pyplot as plt
from classifier import Classifier

class Bagging(Classifier):
    def bagging(self, X, y, test):
        '''
        warm_start reuses the solution of the previous call to fit and add more estimators to the ensemble,
            otherwise, just fit a whole new ensemble.

        n_jobs -1 uses all available cores
        '''
        class_weights = {0: 1, 1: 5}

        bag = ensemble.BaggingClassifier(warm_start=True, n_jobs=-1)

        clf = bag.fit(X, y)

        y_score = bag.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]

        results = bag.predict(test.drop("Class", axis=1).drop("Time", axis=1))

        # Get metrics
        mets = self.compute_metrics(test["Class"], results, y_score)

        print('AUROC:', mets['auroc'])
        print('Accuracy:', mets['accuracy'])
        print('Precision:', mets['precision'])
        print('Recall:', mets['recall'])
        print('F Score:', mets['f'])
        print('Average Precision', mets['ap'])
        print(mets['confusion'])

        # Precision recall measure
        self.plot_precision_recall(test["Class"], y_score, 'Bagging')

        # Plot ROC
        self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Bagging')


def main():
    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1) \
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    # Create train and test groups
    train, test = model_selection.train_test_split(df)

    # X and Y used for sklearn logreg
    X = train.drop("Class", axis=1).drop("Time", axis=1)
    y = train["Class"]

    baggingClassifier = Bagging()

    baggingClassifier.bagging(X, y, test)

if __name__ == '__main__':
    main()
