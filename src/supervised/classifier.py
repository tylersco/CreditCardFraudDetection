import sys
import pandas as pd
from sklearn import metrics
import matplotlib.pyplot as plt

class Classifier:
    def plot_precision_recall(self, y_test, y_score, model_name):
        average_precision = metrics.average_precision_score(y_test, y_score)

        precision, recall, _ = metrics.precision_recall_curve(y_test, y_score)

        plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
        plt.fill_between(recall, precision, step='post', alpha=0.2,
                         color='b')

        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.ylim([0.0, 1.05])
        plt.xlim([0.0, 1.0])
        plt.title('{0} 2-class Precision-Recall curve: AP={1:0.2f}'.format(model_name,
            average_precision))

    def plotROC(self, fpr, tpr, auc, model_name):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = %0.2f)' % auc)
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('{0} ROC'.format(model_name))
        plt.legend(loc="lower right")
        plt.show()

    def compute_metrics(self, y_true, y_pred, y_score):
        fpr, tpr, thresholds = metrics.roc_curve(y_true, y_pred)
        auc = metrics.auc(fpr, tpr)
        precision = metrics.precision_score(y_true, y_pred)
        recall = metrics.recall_score(y_true, y_pred)
        f_score = metrics.f1_score(y_true, y_pred)
        ap_score = metrics.average_precision_score(y_true, y_score)
        accuracy = metrics.accuracy_score(y_true, y_pred)
        confusion = metrics.confusion_matrix(y_true, y_pred)

        mets = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f': f_score,
            'ap': ap_score,
            'auroc': auc,
            'confusion': confusion,
            'fpr': fpr,
            'tpr': tpr
        }

        return mets
