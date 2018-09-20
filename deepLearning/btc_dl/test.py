import tensorflow as tf
W = tf.Variable(
  tf.truncated_normal([5, 5, size_in, size_out],      
      stddev=0.1),
      name="Weights")

B = tf.Variable(tf.constant(0.1, shape=[size_out]), name="Biases")

convolution = tf.nn.conv2d(input, W, strides=[1, 1, 1, 1], padding="SAME")
activation = tf.nn.relu(convolution + B)

tf.nn.max_pool(
activation,
ksize=[1, 2, 2, 1],
strides=[1, 2, 2, 1],
padding="SAME")
