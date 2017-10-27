import os
import pandas as pd
from sklearn import linear_model


class LogisticRegression:

    def logreg(self):
        path = os.path.abspath(os.path.join(os.getcwd(), os.pardir))
        path += '/data/creditcard.csv'

        df = pd.read_csv(path)

        X = df.drop("Class", axis=1).drop("Time", axis=1)

        # According to online sources, LogisticRegression can handle multiple classes ootb
        log_reg_model = linear_model.LogisticRegression()

        log_reg_model.fit(X, df["Class"])

        results = log_reg_model.predict(X)

        for output in results:
            if output != 0:
                print(output)


def main():
    logistic_regression = LogisticRegression()
    logistic_regression.logreg()

if __name__ == '__main__':
    main()