import tensorflow as tf
import numpy as np

def length(tensor):
    s = tensor.get_shape()
    return s[0]

def resp(distance):
    
    rsum = tf.reduce_sum(distance, 0)
    
    res = tf.div(distance, rsum)
    
    return res
    
def getDistance(newd, oldd):
    
    dt = tf.transpose(oldd)
    #Utilizing broadcast to get the distance matrix
    diff = tf.subtract(newd, dt)
    
    #get squared distance or the absolute is really the same for selecting the minimum k
    return tf.exp(tf.multiply(-100.0, tf.square(diff)))

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]

trainD = tf.placeholder(tf.float32, [80, 1])
trainT = tf.placeholder(tf.float32, [80, 1])

dataD = tf.placeholder(tf.float32, [None, 1])

X = np.linspace(0.0, 11.0, num=1000)[:,np.newaxis]

def predict(dl, newD):
    diff = getDistance(newD, trainD)
    prediction = tf.Variable(tf.zeros([dl, 1], dtype=tf.float32), None)
    
    for i in range(0, dl):
        row = diff[i, :]
        respon = resp(row)
        #Get the predicted value
        val = tf.matmul(tf.reshape(respon, [1, 80]), trainT)
        prediction = tf.scatter_add(prediction, [i], val)
    
    return prediction

#training set size is 80, change dl to 80 for training set
dl = 1000
    
pNew = predict(dl, dataD)

init_op = tf.global_variables_initializer()

sess = tf.Session()

sess.run([init_op])


pnew = sess.run(pNew, feed_dict={dataD:X, trainD:trainData, trainT:trainTarget})
print pnew




