import sys
import pandas as pd
from sklearn import metrics, model_selection


from sklearn import tree

class DecisionTree:

    def decisionTree(self, X, y, test):
        decision_tree = tree.DecisionTreeClassifier()
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