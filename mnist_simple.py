from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

w1 = tf.Variable(tf.random_normal(shape=[784, 16]))
w2 = tf.Variable(tf.random_normal(shape=[16, 10]))
b1 = tf.Variable(tf.random_normal(shape=[16]))
b2 = tf.Variable(tf.random_normal(shape=[10]))

h1 = tf.nn.relu(tf.matmul(x, w1) + b1)
y = tf.matmul(h1, w2) + b2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

train_step = tf.train.GradientDescentOptimizer(0.1).minimize(cross_entropy)
print(mnist.train.labels.shape)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    for i in range(10000):
        imgs, labs = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: imgs, y_: labs})
        if i % 500 == 0:
            print(sess.run(cross_entropy, feed_dict={x: imgs, y_: labs}))

    correct_pred = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    print(sess.run(accuracy, feed_dict={y_: mnist.test.labels, x: mnist.test.images}))

    img = cv.imread('2.jpg')
    img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    img = cv.resize(img, (28, 28))
    img = 1 - np.reshape(img, (1, 784)) / 255
    index = tf.argmax(y, 1)
    print(sess.run(index, feed_dict={x: img}))
