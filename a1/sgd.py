import tensorflow as tf
import numpy as np
#import matplotlib.pyplot as plt

# Define dimensions
d = 64     # Size of the parameter space
N = 700   # Number of data sample

with np.load ("./mnist/ECE521Data/tinymnist.npz") as data :
    trainData, trainTarget = data ["x"], data["y"]
    validData, validTarget = data ["x_valid"], data ["y_valid"]
    testData, testTarget = data ["x_test"], data ["y_test"]
    #print testData.shape

# Define placeholders to feed mini_batches
X = tf.placeholder(tf.float32, shape=[None, d])
y_ = tf.placeholder(tf.float32, shape=[None, 1])

# Find values for W that compute y_data = <x, W>
W = tf.Variable(tf.random_uniform([d, 1], -1.0, 1.0))
#We use bias = 0
b = tf.Variable(0.0, name='bias')
y = tf.matmul(X, W, name='y_pred')

#Tune the learning rate, lr
for lr in np.linspace(0.01, 0.25, num=20):
    
    LossOut = []
    #weight decay coefficient
    lam = 1.0
    # Minimize the mean squared errors, panalized by weight.
    
    lossW = tf.div(tf.reduce_mean(
        tf.square(tf.subtract(tf.add(y, b), y_))
            ), 2)
    lossD = tf.multiply(lam/2, tf.reduce_sum(tf.square(W)))
    
    loss = tf.add(lossW, lossD)
    
    optimizer = tf.train.GradientDescentOptimizer(lr)
    train = optimizer.minimize(loss)
    
    # Before starting, initialize the variables
    init = tf.global_variables_initializer()
    
    # Launch the graph.
    sess = tf.Session()
    sess.run(init)
    
    # Fit the line.
    mini_batch_size = 700
    n_batch = N // mini_batch_size # + (N % mini_batch_size != 0) I omit this because it is always divisible
    #Train 501 round or just train over the entire dataset
    for step in range(301):
        i_batch = (step % n_batch)*mini_batch_size
        #print i_batch
        batch = trainData[i_batch:i_batch+mini_batch_size], trainTarget[i_batch:i_batch+mini_batch_size]
        #print sess.run(dify, feed_dict={X: batch[0], y_: batch[1]})
        loss_step, tdummy = sess.run([loss, train], feed_dict={X: batch[0], y_: batch[1]})
        LossOut.append(loss_step)
        #if step == 0 or step == 2000:
            #print(step, sess.run(W))
    #plt.plot(LossOut)
    print LossOut
        
        