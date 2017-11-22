import sys
sys.path.append('../')
import pandas as pd
from sklearn import metrics, model_selection, ensemble
import matplotlib.pyplot as plt
from classifier import Classifier

class Boosting(Classifier):
    def adaboost(self, X, y, valid, test):

        boost = ensemble.AdaBoostClassifier(n_estimators=500)

        clf = boost.fit(X, y)

        # TRAIN DATA

        # y_score = clf.predict_proba(X)[:, 1]
        # results = clf.predict(X)
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

        # y_score = clf.predict_proba(valid.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        # results = clf.predict(valid.drop("Class", axis=1).drop("Time", axis=1))
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

        y_score = clf.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        results = clf.predict(test.drop("Class", axis=1).drop("Time", axis=1))

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
        self.plot_precision_recall(test["Class"], y_score, 'Boosting')

        # Plot ROC
        self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Boosting')


def main():
    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1) \
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    # Create train and test groups
    train, test = model_selection.train_test_split(df, test_size=0.2)
    train, valid = model_selection.train_test_split(train, test_size=0.25)

    # X and Y used for sklearn logreg
    X = train.drop("Class", axis=1).drop("Time", axis=1)
    y = train["Class"]

    adaBoost = Boosting()

    adaBoost.adaboost(X, y, valid, test)

if __name__ == '__main__':
    main()
