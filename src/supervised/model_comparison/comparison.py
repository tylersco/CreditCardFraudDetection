import sys
import pandas as pd
from sklearn import metrics, model_selection, ensemble
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn import naive_bayes


class Comparison:

    def analyze_models(self, X, y, test):

        # model hyperparams
        class_weights = {0: 1, 1: 5}

        # prepare models
        models = []
        models.append(('LogReg', LogisticRegression(class_weight=class_weights)))
        models.append(('Bagging', BaggingClassifier(warm_start=True, n_jobs=-1)))
        models.append(('AdaBoost', AdaBoostClassifier(n_estimators=200)))
        models.append(('DecTree', DecisionTreeClassifier(criterion="gini", splitter="best", min_samples_split=60,
                                                    class_weight=class_weights)))

        seed = 7

        # evaluate each model in turn
        results = []
        precisions = []
        recalls = []
        f_scores = []

        names = []
        scoring = 'accuracy'
        for name, model in models:
            kfold = model_selection.KFold(n_splits=10, random_state=seed)

            model.fit(X, y)

            # Gather relevant stats
            cv_results = model_selection.cross_val_score(model, X, y, cv=kfold, scoring=scoring)
            predictions = model.predict(test.drop("Class", axis=1).drop("Time", axis=1))
            precision = metrics.precision_score(test["Class"], predictions)
            recall = metrics.recall_score(test["Class"], predictions)
            f_score = metrics.f1_score(test["Class"], predictions)

            results.append(cv_results)
            precisions.append(precision)
            recalls.append(recall)
            f_scores.append(f_scores)

            names.append(name)
            msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
            print(name + " precision " + str(precision))
            print(name + " recall " + str(recall))
            print(name + " f_score " + str(f_score))
            print(msg)

        for fig_numb, stats in enumerate([results, precisions, recalls, f_scores]):
            # boxplot algorithm comparison
            fig = plt.figure(fig_numb)
            fig.suptitle('Algorithm Comparison')
            ax = fig.add_subplot(111)
            plt.boxplot(stats)
            ax.set_xticklabels(names)
            plt.show()

def main():
    # Read in data as command line argument
    df = pd.read_csv(sys.argv[1])

    # Drop the attributes deemed useless in our preprocessing/initial analysis
    df = df.drop("V13", axis=1).drop("V15", axis=1).drop("V20", axis=1).drop("V22", axis=1).drop("V23", axis=1) \
        .drop("V24", axis=1).drop("V25", axis=1).drop("V26", axis=1).drop("V28", axis=1)

    df = df.head(10000)
    # Create train and test groups
    train, test = model_selection.train_test_split(df)

    # X and Y used for sklearn logreg
    X = train.drop("Class", axis=1).drop("Time", axis=1)
    y = train["Class"]

    comparison = Comparison()

    comparison.analyze_models(X, y, test)


if __name__ == '__main__':
    main()