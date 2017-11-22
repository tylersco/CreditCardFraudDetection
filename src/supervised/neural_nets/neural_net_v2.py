import sys
sys.path.append('../../data/')

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics

from load_data import load_data

I = 20
H1 = 64
H2 = 64
O = 1

X = tf.placeholder(tf.float32, [None, I])

W1 = tf.Variable(tf.truncated_normal([I, H1]))
B1 = tf.Variable(tf.truncated_normal([O]))
W2 = tf.Variable(tf.truncated_normal([H1, H2]))
B2 = tf.Variable(tf.truncated_normal([H2]))
W3 = tf.Variable(tf.truncated_normal([H2, O]))
B3 = tf.Variable(tf.truncated_normal([O]))

# Model
Y1 = tf.nn.tanh(tf.matmul(X, W1) + B1)
Y2 = tf.nn.tanh(tf.matmul(Y1, W2) + B2)
Y = tf.matmul(Y2, W3) + B3

Y_pred = tf.nn.sigmoid(Y, name='Y_pred')
Y_pred_round = tf.round(Y_pred, name='Y_pred_round')

# Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, O])

# Different evaluation metrics
is_correct = tf.equal(Y_pred_round, Y_)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
roc_auc = tf.metrics.auc(Y_, Y_pred, num_thresholds=200, name='roc_auc')
recall = tf.metrics.recall(Y_, Y_pred_round)
precision = tf.metrics.precision(Y_, Y_pred_round)
#pr_auc = tf.metrics.auc(Y_, Y_pred, curve='PR', num_thresholds=200, name='pr_auc')
#confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.squeeze(Y_), predictions=tf.squeeze(Y_pred_round), num_classes=2, name='confusion_matrix')

lr = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Y, targets=Y_, pos_weight=8))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

init = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()

with tf.Session() as sess:

    # Read in data from each class and create train/test splits
    x_genuine, y_genuine, x_fraudulent, y_fraudulent = load_data(sys.argv[1], [0, 13, 15, 20, 22, 23, 24, 25, 26, 28])

    x_gen_train, x_gen_test, y_gen_train, y_gen_test = train_test_split(x_genuine, y_genuine, test_size=0.2)
    x_gen_train, x_gen_valid, y_gen_train, y_gen_valid = train_test_split(x_gen_train, y_gen_train, test_size=0.25)

    x_fra_train, x_fra_test, y_fra_train, y_fra_test = train_test_split(x_fraudulent, y_fraudulent, test_size=0.2)
    x_fra_train, x_fra_valid, y_fra_train, y_fra_valid = train_test_split(x_fra_train, y_fra_train, test_size=0.25)

    x_train = np.concatenate((x_gen_train, x_fra_train))
    y_train = np.concatenate((y_gen_train, y_fra_train))

    x_valid = np.concatenate((x_gen_valid, x_fra_valid))
    y_valid = np.concatenate((y_gen_valid, y_fra_valid))

    x_test = np.concatenate((x_gen_test, x_fra_test))
    y_test = np.concatenate((y_gen_test, y_fra_test))

    learning_rate = 0.005
    epochs = 300
    minibatch_size = 256

    sess.run(init)
    sess.run(init2)

    for epoch in range(epochs):
        # Shuffle the data
        shuffle = np.random.permutation(len(y_train))
        x_train, y_train = x_train[shuffle], y_train[shuffle]

        for i in range(0, len(y_train), minibatch_size):
            x_train_mb, y_train_mb = x_train[i:i + minibatch_size], y_train[i:i + minibatch_size]

            train_data = {X: x_train_mb, Y_: y_train_mb, lr: learning_rate}

            # Train
            sess.run(optimizer, feed_dict=train_data)

        if epoch % 50 == 0:
            train_data = {X: x_train, Y_: y_train}
            acc, c, auroc, rec, prec  = sess.run([accuracy, cost, roc_auc, recall, precision], feed_dict=train_data)
            pred, pred_round = sess.run([Y_pred, Y_pred_round], feed_dict=train_data)
            average_precision = metrics.average_precision_score(y_train, pred)
            f_score = metrics.f1_score(y_train, pred_round)
            print('Average Precision:', average_precision)
            print('F Score:', f_score)
            print('Train accuracy: {0}, Cost: {1}, AUROC: {2}, Recall: {3}, Precision: {4}'.format(acc, c, auroc, rec, prec) + '\n')

        if epoch % 50 == 0:
            valid_data = {X: x_valid, Y_: y_valid}
            acc, c, auroc, rec, prec = sess.run([accuracy, cost, roc_auc, recall, precision], feed_dict=valid_data)
            pred, pred_round = sess.run([Y_pred, Y_pred_round], feed_dict=valid_data)
            average_precision = metrics.average_precision_score(y_valid, pred)
            f_score = metrics.f1_score(y_valid, pred_round)
            print('Average Precision:', average_precision)
            print('F Score:', f_score)
            print('Valid accuracy: {0}, Cost: {1}, AUROC: {2}, Recall: {3}, Precision: {4}'.format(acc, c, auroc, rec, prec) + '\n')

    valid_data = {X: x_valid, Y_: y_valid}
    acc, c, auroc  = sess.run([accuracy, cost, roc_auc], feed_dict=valid_data)
    pred, pred_round = sess.run([Y_pred, Y_pred_round], feed_dict=valid_data)
    average_precision = metrics.average_precision_score(y_valid, pred)
    f_score = metrics.f1_score(y_valid, pred_round)
    print('Average Precision:', average_precision)
    print('F Score:', f_score)
    print('Valid accuracy: {0}, Cost: {1}, AUROC: {2}, Recall: {3}, Precision: {4}'.format(acc, c, auroc, rec, prec) + '\n')

    test_data = {X: x_test, Y_: y_test}
    acc, c, auroc, rec, prec = sess.run([accuracy, cost, roc_auc, recall, precision], feed_dict=test_data)
    pred, pred_round = sess.run([Y_pred, Y_pred_round], feed_dict=test_data)
    average_precision = metrics.average_precision_score(y_test, pred)
    f_score = metrics.f1_score(y_test, pred_round)
    print('Average Precision:', average_precision)
    print('F Score:', f_score)
    print('Test accuracy: {0}, Cost: {1}, AUROC: {2}, Recall: {3}, Precision: {4}'.format(acc, c, auroc, rec, prec) + '\n')

    precision, recall, _ = metrics.precision_recall_curve(y_test, pred)
    plt.step(recall, precision, color='b', alpha=0.2,
                 where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')

    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Neural Network 2-class Precision-Recall curve: AP={0:0.2f}'.format(
        average_precision))

    fpr, tpr, _ = metrics.roc_curve(y_test, pred)
    plt.figure()
    lw = 2
    plt.plot(fpr, tpr, color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % auroc[0])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Neural Network ROC')
    plt.legend(loc="lower right")
    plt.show()
