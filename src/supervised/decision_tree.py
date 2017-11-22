import sys
import pandas as pd
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt
from classifier import Classifier
from sklearn import tree

class DecisionTree(Classifier):
    def decisionTree(self, X, y, valid, test):

        '''
            setting criterion to entropy decreased the model precision recall score to 54 percent.

            setting splitter to random decreased the model precision recall score to 53 percent.

            setting min_samples to 4 decreased the model pr score, however raising it past 8 seems to keep
                PR above 55 percent.
        '''

        class_weights = {0: 1, 1: 8}

        decision_tree = tree.DecisionTreeClassifier(criterion="gini", splitter="best", min_samples_split=125,
                                                    class_weight=class_weights)

        clf = decision_tree.fit(X, y)

        # TRAIN DATA

        # y_score = decision_tree.predict_proba(X)[:, 1]
        # results = decision_tree.predict(X)
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

        # y_score = decision_tree.predict_proba(valid.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        # results = decision_tree.predict(valid.drop("Class", axis=1).drop("Time", axis=1))
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

        y_score = decision_tree.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]
        results = decision_tree.predict(test.drop("Class", axis=1).drop("Time", axis=1))

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
        self.plot_precision_recall(test["Class"], y_score, 'Decision Tree')

        # Plot ROC
        self.plotROC(mets['fpr'], mets['tpr'], mets['auroc'], 'Decision Tree')


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

    decision_tree = DecisionTree()

    decision_tree.decisionTree(X, y, valid, test)

if __name__ == '__main__':
    main()
