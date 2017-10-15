import sys
sys.path.append('../data/')

import tensorflow as tf
import numpy as np

from sklearn.model_selection import train_test_split

from load_data import load_data

I = 30
H1 = 64
H2 = 64
O = 1

X = tf.placeholder(tf.float32, [None, I])

W1 = tf.Variable(tf.truncated_normal([I, H1]))
B1 = tf.Variable(tf.zeros([H1]))
W2 = tf.Variable(tf.truncated_normal([H1, H2]))
B2 = tf.Variable(tf.zeros([H2]))
W3 = tf.Variable(tf.truncated_normal([H2, O]))
B3 = tf.Variable(tf.zeros([O]))

# Model
Y1 = tf.nn.relu(tf.matmul(X, W1) + B1)
Y2 = tf.nn.relu(tf.matmul(Y1, W2) + B2)
Y = tf.matmul(Y1, W3) + B3

# Placeholder for correct answers
Y_ = tf.placeholder(tf.float32, [None, O])

is_correct = tf.equal(tf.round(tf.nn.sigmoid(Y)), Y_)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

lr = tf.placeholder(tf.float32)
cost = tf.reduce_mean(tf.nn.weighted_cross_entropy_with_logits(logits=Y, targets=Y_, pos_weight=0.7))
optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(cost)

init = tf.global_variables_initializer()

with tf.Session() as sess:

    # Read in data from each class and create train/test splits
    x_genuine, y_genuine, x_fraudulent, y_fraudulent = load_data(sys.argv[1])

    x_gen_train, x_gen_test, y_gen_train, y_gen_test = train_test_split(x_genuine, y_genuine, test_size=0.2)
    x_fra_train, x_fra_test, y_fra_train, y_fra_test = train_test_split(x_fraudulent, y_fraudulent, test_size=0.2)

    x_train = np.concatenate((x_gen_train, x_fra_train))
    y_train = np.concatenate((y_gen_train, y_fra_train))

    x_test = np.concatenate((x_gen_test, x_fra_test))
    y_test = np.concatenate((y_gen_test, y_fra_test))

    learning_rate = 0.00001
    epochs = 150
    minibatch_size = 1024

    sess.run(init)

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
        acc, c = sess.run([accuracy, cost], feed_dict=train_data)
        print('Train accuracy: {0}, Cost: {1}'.format(acc, c))


    #test_data = {X: X_test_arr, Y_: y_test_arr}
    #acc, se = sess.run([accuracy, squared_error], feed_dict=test_data)
    #print('\nTest accuracy: {0}, Mean Squared Error: {1}'.format(acc, se))
