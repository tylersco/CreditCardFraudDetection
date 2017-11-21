import sys
import pandas as pd
from sklearn import linear_model, metrics, model_selection
from sklearn.naive_bayes import GaussianNB as GNB
import matplotlib.pyplot as plt
from classifier import Classifier

class NaiveBayes(Classifier):
    def bayes(self, X, y, test):
        nb_model = GNB()
        nb_model.fit(X, y)

        #y_score = log_reg_model.decision_function(test.drop("Class", axis=1).drop("Time", axis=1))
        y_score = nb_model.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        results = nb_model.predict(test.drop("Class", axis=1).drop("Time", axis=1))

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
        self.plot_precision_recall(test["Class"], y_score, 'Naive Bayes')

        # Plot ROC
        self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Naive Bayes')


def main():

    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1)\
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    # Create train and test groups
    train, test = model_selection.train_test_split(df)

    # X and Y used for sklearn logreg
    X = train.drop("Class", axis=1).drop("Time", axis=1)
    y = train["Class"]

    nb = NaiveBayes()

    nb.bayes(X, y, test)

if __name__ == '__main__':
    main()
