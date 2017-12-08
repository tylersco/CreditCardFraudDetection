'''
Version 1 of the neural network
'''

import sys
sys.path.append('../../data/')

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

from load_data import load_data

# Input neurons
I = 30
# Hidden layer 1 neurons
H1 = 64
# Hidden layer 2 neurons
H2 = 64
# Output neurons
O = 1

X = tf.placeholder(tf.float32, [None, I])

# Weights and biases associated with each layer
W1 = tf.Variable(tf.truncated_normal([I, H1]))
B1 = tf.Variable(tf.zeros([H1]))
W2 = tf.Variable(tf.truncated_normal([H1, H2]))
B2 = tf.Variable(tf.zeros([H2]))
W3 = tf.Variable(tf.truncated_normal([H2, O]))
B3 = tf.Variable(tf.zeros([O]))

# Model
# Feedforward pass through network
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y = tf.matmul(Y1, W3) + B3

Y_pred = tf.nn.sigmoid(Y, name='Y_pred')
Y_pred_round = tf.round(Y_pred, name='Y_pred_round')

# Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, O])

# Different evaluation metrics
is_correct = tf.equal(Y_pred_round, Y_)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32), name='accuracy')
roc_auc = tf.metrics.auc(Y_, Y_pred, name='roc_auc')
pr_auc = tf.metrics.auc(Y_, Y_pred, curve='PR', name='pr_auc')
confusion_matrix = tf.contrib.metrics.confusion_matrix(labels=tf.squeeze(Y_), predictions=tf.squeeze(Y_pred_round), num_classes=2, name='confusion_matrix')

# Loss function and associated optimizer
lr = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Y, targets=Y_, pos_weight=0.95))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

init = tf.global_variables_initializer()
init2 = tf.local_variables_initializer()

saver = tf.train.Saver()

with tf.Session() as sess:

    # Read in data from each class and create train/test splits
    x_genuine, y_genuine, x_fraudulent, y_fraudulent = load_data(sys.argv[1])

    x_gen_train, x_gen_test, y_gen_train, y_gen_test = train_test_split(x_genuine, y_genuine, test_size=0.2)
    x_gen_train, x_gen_valid, y_gen_train, y_gen_valid = train_test_split(x_gen_train, y_gen_train, test_size=0.25)

    x_fra_train, x_fra_test, y_fra_train, y_fra_test = train_test_split(x_fraudulent, y_fraudulent, test_size=0.2)
    x_fra_train, x_fra_valid, y_fra_train, y_fra_valid = train_test_split(x_fra_train, y_fra_train, test_size=0.25)

    # Split datasets

    x_train = np.concatenate((x_gen_train, x_fra_train))
    y_train = np.concatenate((y_gen_train, y_fra_train))

    x_valid = np.concatenate((x_gen_valid, x_fra_valid))
    y_valid = np.concatenate((y_gen_valid, y_fra_valid))

    x_test = np.concatenate((x_gen_test, x_fra_test))
    y_test = np.concatenate((y_gen_test, y_fra_test))

    learning_rate = 0.00001
    epochs = 500
    minibatch_size = 1024

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

        train_data = {X: x_train, Y_: y_train}
        acc, c, auroc, auprc, cf_matrix  = sess.run([accuracy, cost, roc_auc, pr_auc, confusion_matrix], feed_dict=train_data)
        print('Train accuracy: {0}, Cost: {1}, AUROC: {2}, AUPRC: {3}'.format(acc, c, auroc, auprc))
        print(cf_matrix, '\n')

        if epoch % 10 == 0:
            valid_data = {X: x_valid, Y_: y_valid}
            acc, c, auroc, auprc, cf_matrix  = sess.run([accuracy, cost, roc_auc, pr_auc, confusion_matrix], feed_dict=valid_data)
            print('Valid accuracy: {0}, Cost: {1}, AUROC: {2}, AUPRC: {3}'.format(acc, c, auroc, auprc))
            print(cf_matrix, '\n')

    valid_data = {X: x_valid, Y_: y_valid}
    acc, c, auroc, auprc, cf_matrix  = sess.run([accuracy, cost, roc_auc, pr_auc, confusion_matrix], feed_dict=valid_data)
    print('Valid accuracy: {0}, Cost: {1}, AUROC: {2}, AUPRC: {3}'.format(acc, c, auroc, auprc))
    print(cf_matrix, '\n')

    test_data = {X: x_test, Y_: y_test}
    acc, c, auroc, auprc, cf_matrix  = sess.run([accuracy, cost, roc_auc, pr_auc, confusion_matrix], feed_dict=test_data)
    print('Test accuracy: {0}, Cost: {1}, AUROC: {2}, AUPRC: {3}'.format(acc, c, auroc, auprc))
    print(cf_matrix, '\n')

    save_path = saver.save(sess, '../../models/neural_net_v1/neural_net_v1')
    print("Model saved in file: %s" % save_path)
