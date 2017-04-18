#-*- coding:utf-8 -*-

from __future__ import print_function

import tensorflow as tf

# Import MNIST data
from tensorflow.examples.tutorials.mnist import input_data

x_data = [
    [0, 1, 0,
     0, 1, 0,
     0, 1, 0],
    [1, 0, 1,
     0, 1, 0,
     1, 0, 1],
    [1, 1, 1,
     0, 1, 0,
     1, 1, 1],
    [1, 0, 1,
     1, 0, 1,
     1, 1, 1],
    [1, 0, 1,
     1, 1, 1,
     1, 0, 1]]

'''       I,X,Z,U,H'''
y_data = [
    [1,0,0,0,0],
    [0,1,0,0,0],
    [0,0,1,0,0],
    [0,0,0,1,0],
    [0,0,0,0,1]
          ]



# Parameters
learning_rate = 0.001
training_epochs = 30000
batch_size = 100
display_step = 1

# tf Graph Input
x = tf.placeholder(tf.float32, [None, 9]) # mnist data image of shape 28*28=784
y = tf.placeholder(tf.float32, [None, 5]) # 0-9 digits recognition => 10 classes

# Set model weights
W = tf.Variable(tf.zeros([9, 5]))
b = tf.Variable(tf.zeros([5]))

# Construct model
pred = tf.nn.softmax(tf.matmul(x, W) + b) # Softmax

# Minimize error using cross entropy
cost = tf.reduce_mean(-tf.reduce_sum(y*tf.log(pred), reduction_indices=1))
# Gradient Descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        _, c = sess.run([optimizer, cost], feed_dict={x: x_data,
                                                      y: y_data})

        print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c)," W =",sess.run(W),"B =",sess.run(b))

    print("Optimization Finished!")

    x_test = [
        [0, 1, 0,
        0, 1, 0,
        0, 0, 0],
        [1, 0, 1,
        0, 1, 0,
        1, 0, 1]]
    y_test = [
        [1, 0, 0, 0, 0],
        [0, 1, 0, 0, 0]
    ]
    # Test model
    correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
    # Calculate accuracy
    p = sess.run(pred, feed_dict={x:x_test,y:y_test})
    print("예측값:",p)

    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print("Accuracy:", accuracy.eval({x: x_test, y: y_test}))
