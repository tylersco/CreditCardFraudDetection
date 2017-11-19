import sys
import pandas as pd
from sklearn import metrics, model_selection
import matplotlib.pyplot as plt

from sklearn import tree

class DecisionTree:
    def plot_precision_recall(self, y_test, y_score):
        average_precision = metrics.average_precision_score(y_test, y_score)

        print('Average precision-recall score: {0:0.2f}'.format(
            average_precision))

        precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('Decision Tree 2-class Precision-Recall curve: AP={0:0.2f}'.format(
            average_precision))

    def plotROC(self, fpr, tpr, auc):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Decision Tree ROC')
        plt.legend(loc="lower right")
        plt.show()

    def decisionTree(self, X, y, test):

        '''
            setting criterion to entropy decreased the model precision recall score to 54 percent.

            setting splitter to random decreased the model precision recall score to 53 percent.

            setting min_samples to 4 decreased the model pr score, however raising it past 8 seems to keep
                PR above 55 percent.
        '''

        class_weights = {0: 1, 1: 5}

        decision_tree = tree.DecisionTreeClassifier(criterion="gini", splitter="best", min_samples_split=60,
                                                    class_weight=class_weights)

        clf = decision_tree.fit(X, y)

        # y_score = log_reg_model.decision_function(test.drop("Class", axis=1).drop("Time", axis=1))
        y_score = decision_tree.predict_proba(test.drop("Class", axis=1).drop("Time", axis=1))[:, 1]

        results = decision_tree.predict(test.drop("Class", axis=1).drop("Time", axis=1))
        accuracy = decision_tree.score(test.drop("Class", axis=1).drop("Time", axis=1), test["Class"])
        confusion = metrics.confusion_matrix(test["Class"], results)

        # AUC and ROC measures
        fpr, tpr, thresholds = metrics.roc_curve(test["Class"], results)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(test["Class"], results)
        recall = metrics.recall_score(test["Class"], results)
        f_score = metrics.f1_score(test["Class"], results)

        print('AUROC:', auc)
        print('Accuracy:', accuracy)
        print('Precision:', precision)
        print('Recall:', recall)
        print('F Score:', f_score)

        # Precision recall measure
        self.plot_precision_recall(test["Class"], y_score)

        # Plot ROC
        self.plotROC(fpr, tpr, auc)

        return results, accuracy, confusion


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

    decision_tree = DecisionTree()

    total = decision_tree.decisionTree(X, y, test)
    print(total[0])
    print(total[1])
    print(total[2])


if __name__ == '__main__':
    main()