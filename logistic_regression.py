#-*- coding:utf-8 -*-

import focus.TrainData as td
import datetime as dt
import tensorflow as tf
import numpy as np
import random

# Parameters
learning_rate = 0.01
batch_size = 100
n_classes = [539,53,15890]
t_classes = 16482#-15890
n_layers = 512

def main():

    train_data = td.TrainData("/Users/sk/Documents/data/train/news_train.txt",n_classes)
    test_data = td.TrainData("/Users/sk/Documents/data/train/news_test.txt", n_classes)

    # tf Graph Input
    X = tf.placeholder(tf.float32, [None, t_classes])  # mnist data image of shape 28*28=784
    Y = tf.placeholder(tf.float32, [None, 2])  # 0-9 digits recognition => 10 classes

    # Set model weights
    W = tf.Variable(tf.zeros([t_classes, 2]))
    b = tf.Variable(tf.zeros([2]))

    # Construct model
    pred = tf.nn.softmax(tf.matmul(X, W) + b)  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(Y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())


        while train_data.has_fetch_data():
            batch_x,batch_y  = train_data.fetch_train_data(100)
            #batch_y = [np.array([1, 0]), ] * len(batch_x)
            _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
            print("cost=", "{:.9f}".format(c))

        print("Optimization Finished!")

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(Y, 1))
        # Calculate accuracy
        test_xs, test_ys = test_data.fetch_all_data()

        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        print("Accuracy:", accuracy.eval({X: test_xs, Y: test_ys}))





if __name__ == "__main__":
    main()