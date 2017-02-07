import tensorflow as tf

dl = tf.placeholder(tf.int32, None)

zeros = tf.zeros(dl, dtype=tf.float32)

sess = tf.Session()
print sess.run(zeros, feed_dict={dl:[80, 1]})