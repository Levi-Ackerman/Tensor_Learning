import tensorflow as tf

# 矩阵定义
a = tf.constant([[1.0, 2.0]])
b = tf.constant([[3.0], [4.0]])

# 矩阵相乘，只有计算图，不执行
result = tf.matmul(a, b)
print(result)

# 执行计算
with tf.Session() as sess:
    # 运算出result的结果
    print(sess.run(result))
    # 生成3*2的全0矩阵和全1矩阵相加
    print(sess.run(tf.zeros([3, 2]) + tf.ones([3, 2])))
    # 定值矩阵
    print(sess.run(tf.fill([1], 5.5)))
    # 生成3*2的随机数矩阵，随机算法为均值为0，标准差为1的正态分布（高斯分布）
    print(sess.run(tf.random_normal([3, 2], 0, 1)))
    # 随机一个平均分布的矩阵
    print(sess.run(tf.random_uniform([3, 2])))
    # 截断正态分布（按正态分布随机，随到的数离均值大于2倍方差时，将重新随机）
    print(sess.run(tf.truncated_normal([3, 2], 0, 1)))


    # 生成权重参数
    w = tf.Variable(a)
    print(w)

se = tf.InteractiveSession()
print(a.eval())
se.close()