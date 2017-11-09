import os
import pandas as pd
from sklearn import linear_model, metrics, model_selection


class LogisticRegression:

    def logreg(self, X, y, test):

        # Associate higher weight (of 2) with the positive class:
        weights = {1: 2, 0: 1}

        # According to online sources, LogisticRegression can handle multiple classes ootb
        log_reg_model = linear_model.LogisticRegression(class_weight=weights)

        log_reg_model.fit(X, y)

        results = log_reg_model.predict(test.drop("Class", axis=1).drop("Time", axis=1))

        print(results)

        accuracy = log_reg_model.score(test.drop("Class", axis=1).drop("Time", axis=1), test["Class"])

        confusion = metrics.confusion_matrix(test["Class"], results)

        return results, accuracy, confusion


def main():
    # Create dataframe
    path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
    path += '/data/creditcard.csv'
    df = pd.read_csv(path)

    #Create train and test groups
    train, test = model_selection.train_test_split(df)

    # X and Y used for sklearn logreg
    X = train.drop("Class", axis=1).drop("Time", axis=1)
    y = train["Class"]

    logistic_regression = LogisticRegression()

    total = logistic_regression.logreg(X, y, test)
    print(total[0])
    print(total[1])
    print(total[2])


if __name__ == '__main__':
    main()