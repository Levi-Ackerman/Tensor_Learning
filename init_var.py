# coding:utf-8
import tensorflow as tf

#输入层x，1层隐藏层a，输出层out，两层权重w1，w2

# 输入矩阵 1*2
x = tf.constant([[0.4, 0.3]])

# 第1层权重初始化 2*3
w1 = tf.Variable(tf.random_normal([2, 3]))

# 输出层权重初始化 3*1
w2 = tf.Variable(tf.random_normal([3, 1]))

# 前向传播计算图
a = tf.matmul(x, w1) #第1层
out = tf.matmul(a, w2) #第2层

with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    result = sess.run(out)
    print("result is ", result)

