import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def length(tensor):
    s = tensor.get_shape()
    return s[0]

def resp(distance, k):
    
    nd = tf.mul(distance, -1)
    
    values, indices = tf.nn.top_k(nd, k, False, None)
    
    l = length(distance)
    
    base = tf.Variable(tf.zeros(l, dtype=tf.float32), None)
    
    update = tf.ones(k, tf.float32)
    
    res_one = tf.scatter_add(base, indices, update, None, None)
    
    res = tf.div(res_one, k)
    
    return res
    
def getDistance(newd, oldd):
    
    dt = tf.transpose(oldd)
    
    diff = tf.subtract(newd, dt)
    
    return tf.abs(diff)

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

X = np.linspace(0.0, 11.0, num=1000)[:,np.newaxis]

trainD = tf.placeholder(tf.float32, [80, 1])
trainT = tf.placeholder(tf.float32, [80, 1])
newD = tf.placeholder(tf.float32, [None, 1])
newT = tf.placeholder(tf.float32, [None, 1])

for k in [1]: #, 3, 5, 50
    dl = 1000
    #tf.constant([length(newD)])
    #
    #print validTarget
    #print "Now the distance matrix\n"
    
    diff = getDistance(newD, trainD)
    prediction = tf.Variable(tf.zeros([dl, 1], dtype=tf.float32), None)
    
    for i in range(0, dl):
        row = diff[i, :]
        respon = resp(row, k)
        val = tf.matmul(tf.reshape(respon, [1, 80]), trainT)
        prediction = tf.scatter_add(prediction, [i], val)
        
    finalp = prediction
    
    #mse = tf.square(tf.subtract(newT, prediction))
    
    #loss = tf.div(tf.reduce_sum(mse, 0), 2*dl)
    
    init_op = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    sess.run([init_op])
    pred = sess.run(finalp, feed_dict={newD:X, trainD:trainData, trainT:trainTarget})
    plt.plot(pred)
    plt.show()
    #print pred.transpose()
    
    sess.close()






