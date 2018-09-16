# coding:utf-8
# 利用占位作为输入层，每次用真实输入填充占位，实现计算

import tensorflow as tf

x = tf.placeholder(float, [1, 2])

w1 = tf.Variable(tf.random_normal([2, 3]))
w2 = tf.Variable(tf.random_normal([3, 1]))

a = tf.matmul(x, w1)
out = tf.matmul(a, w2)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(out, {x: [[0.4, 0.3]]})
    print("placeholder result is :", result)
