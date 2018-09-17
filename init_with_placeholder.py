# coding:utf-8
# 利用占位作为输入层，每次用真实输入填充占位，实现计算

import tensorflow as tf

x = tf.placeholder(float, [None, 2])

w1 = tf.Variable(tf.random_normal([2, 1]))

out = tf.matmul(x, w1)

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(out, {x: [[0.4, 0.3], [0.8, 0.6]]})
    print("placeholder result is :", result)

    print("w1 is ", sess.run(w1))
