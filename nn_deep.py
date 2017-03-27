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


    # Create the model
    X = tf.placeholder(tf.float32, [None, t_classes])
    Y = tf.placeholder(tf.float32, [None, 2])

    #W = tf.get_variable("W", shape=[t_classes, 2],initializer=tf.contrib.layers.xavier_initializer())
    #b = tf.Variable(tf.zeros([2]))

    W1 = tf.get_variable("W1", shape=[t_classes, n_layers],
                         initializer=tf.contrib.layers.xavier_initializer())

    b1 = tf.Variable(tf.random_normal([n_layers]))
    L1 = tf.nn.relu(tf.matmul(X, W1) + b1)

    W2 = tf.get_variable("W2", shape=[n_layers, n_layers],
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.Variable(tf.random_normal([n_layers]))
    L2 = tf.nn.relu(tf.matmul(L1, W2) + b2)

    W3 = tf.get_variable("W3", shape=[n_layers, n_layers],
                         initializer=tf.contrib.layers.xavier_initializer())
    b3 = tf.Variable(tf.random_normal([n_layers]))
    L3 = tf.nn.relu(tf.matmul(L2, W3) + b3)

    W4 = tf.get_variable("W4", shape=[n_layers, n_layers],
                         initializer=tf.contrib.layers.xavier_initializer())
    b4 = tf.Variable(tf.random_normal([n_layers]))
    L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)

    W5 = tf.get_variable("W5", shape=[n_layers, 2],
                         initializer=tf.contrib.layers.xavier_initializer())

    b5 = tf.Variable(tf.random_normal([2]))

    hypothesis = tf.matmul(L4, W5) + b5

    # define cost/loss & optimizer
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))

    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

    sess = tf.Session()
    sess.run(tf.global_variables_initializer())

    while train_data.has_fetch_data():
        batch_x,batch_y  = train_data.fetch_train_data(100)
        #batch_y = [np.array([1, 0]), ] * len(batch_x)
        _, c = sess.run([optimizer, cost], feed_dict={X: batch_x, Y: batch_y})
        print("cost=", "{:.9f}".format(c))



    print("Optimization Finished!")

    # Test model and check accuracy
    correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    test_xs,test_ys = test_data.fetch_all_data()
    #test_ys = [np.array([1, 0]), ] * len(test_xs)

    print('Accuracy:', sess.run(accuracy, feed_dict={
        X: test_xs, Y: test_ys}))




if __name__ == "__main__":
    main()