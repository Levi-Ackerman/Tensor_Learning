import tensorflow as tf

tf.enable_eager_execution()

a = tf.constant([[1, 2]])

print(tf.matmul(a, tf.matrix_transpose(a)))

print(tf.add(a,1))

