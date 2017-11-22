import sys
import pandas as pd
from sklearn import linear_model, metrics, model_selection
import matplotlib.pyplot as plt
from classifier import Classifier

class LogisticRegression(Classifier):
    def logreg(self, X, y, valid, test):
        '''
        The “balanced” mode uses the values of y to automatically adjust weights inversely proportional to class
            frequencies in the input data as n_samples / (n_classes * np.bincount(y)).
        '''
        class_weights = {0: 1, 1: 8}
        log_reg_model = linear_model.LogisticRegression(class_weight=class_weights, penalty='l2', C=10.0, max_iter=500)
        log_reg_model.fit(X, y)

        # TRAIN DATA

        # y_score = log_reg_model.predict_proba(X)[:, 1]
        # results = log_reg_model.predict(X)
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

        # y_score = log_reg_model.predict_proba(valid.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        # results = log_reg_model.predict(valid.drop("Class", axis=1).drop("Time", axis=1))
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

        y_score = log_reg_model.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        results = log_reg_model.predict(test.drop("Class", axis=1).drop("Time", axis=1))

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
        self.plot_precision_recall(test["Class"], y_score, 'Logistic Regression')

        # Plot ROC
        self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Logistic Regression')


def main():

    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1)\
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    # Create train and test groups
    train, test = model_selection.train_test_split(df, test_size=0.2)
    train, valid = model_selection.train_test_split(train, test_size=0.25)

    # X and Y used for sklearn logreg
    X = train.drop("Class", axis=1).drop("Time", axis=1)
    y = train["Class"]

    logistic_regression = LogisticRegression()

    logistic_regression.logreg(X, y, valid, test)

if __name__ == '__main__':
    main()
