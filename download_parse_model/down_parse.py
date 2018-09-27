import tensorflow as tf

tf.enable_eager_execution()

url = "http://download.tensorflow.org/data/iris_training.csv"

fp = tf.keras.utils.get_file(fname="cache", origin=url)


def parse_csv(line):
    example_defaults = [[0.], [0.], [0.], [0.], [0]]
    parsed_line = tf.decode_csv(line, example_defaults)
    f = tf.reshape(parsed_line[:-1], shape=(4,))
    l = tf.reshape(parsed_line[-1], shape=(1,))
    return f, l


train_data = tf.data.TextLineDataset(fp)
train_data = train_data.skip(1)
train_data = train_data.map(parse_csv)
train_data = train_data.shuffle(buffer_size=1000)
train_data = train_data.batch(32)

features, label = iter(train_data).next()
print("exampe features: ", features[0])
print("example label: ", label[0])
