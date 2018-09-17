# coding:utf-8
import tensorflow as tf
import numpy as np

# mini batch 数量
MINI_BATCH_SIZE = 8
# 迭代次数
STEP_COUNT = 10000
# 训练数据个数
TRAIN_COUNT = 32

# 0. 准备训练数据和测试数据
X = np.random.rand(TRAIN_COUNT, 2)
# print(input_data)

# input data中，两个维度的和大于1时，Y结果赋值为1
Y = [[int(a + b > 1)] for (a, b) in X]
# print(Y)

test_x = np.random.rand(TRAIN_COUNT, 2)
test_y = [[int(a + b > 1)] for (a, b) in test_x]

# 1. 定义参数
x = tf.placeholder(tf.float32, [None, 2])
y_ = tf.placeholder(tf.float32, [None, 1])

w1 = tf.Variable(tf.random_normal([2, 3]))
w2 = tf.Variable(tf.random_normal([3, 1]))

y = tf.matmul(tf.matmul(x, w1), w2)

# 2. 定义损失函数，学习率，BP算法
loss = tf.reduce_mean(tf.square(y_ - y))
train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(loss)

# 3. 训练模型
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())
    print("Before train: w1 is ", sess.run(w1), "; w2 is ", sess.run(w2))

    for i in range(STEP_COUNT):
        start = i * MINI_BATCH_SIZE % TRAIN_COUNT
        end = start + MINI_BATCH_SIZE
        # print(input_data[start:end])
        sess.run(train_step,
                 feed_dict={x: X[start:end], y_: Y[start:end]})
        if(i % 500 == 0):
            print("loss value is ", sess.run(loss, {x: X, y_: Y}))

    print("After train: w1 is ", sess.run(w1), "; w2 is ", sess.run(w2))

    for i in range(TRAIN_COUNT):
        result = sess.run(y, feed_dict={x: [test_x[i]]})
        print("预测值：train_step:", result, " 真实值：", test_y[i])
