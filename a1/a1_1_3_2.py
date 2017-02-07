import tensorflow as tf
import numpy as np

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
    #Utilizing broadcast to get the distance matrix
    diff = tf.subtract(newd, dt)
    
    return tf.abs(diff)

np.random.seed(521)
Data = np.linspace(1.0 , 10.0 , num =100) [:, np. newaxis]
Target = np.sin( Data ) + 0.1 * np.power( Data , 2) \
+ 0.5 * np.random.randn(100 , 1)
randIdx = np.arange(100)
np.random.shuffle(randIdx)
trainData, trainTarget  = Data[randIdx[:80]], Target[randIdx[:80]]
#validData, validTarget = Data[randIdx[80:90]], Target[randIdx[80:90]]
#testData, testTarget = Data[randIdx[90:100]], Target[randIdx[90:100]]

trainD = tf.placeholder(tf.float32, [80, 1])
trainT = tf.placeholder(tf.float32, [80, 1])
#testD = tf.placeholder(tf.float32, [10, 1])
#testT = tf.placeholder(tf.float32, [10, 1])
#validD = tf.placeholder(tf.float32, [10, 1])
#validT = tf.placeholder(tf.float32, [10, 1])
dataD = tf.placeholder(tf.float32, [None, 1])

X = np.linspace(0.0, 11.0, num=1000)[:,np.newaxis]

def predict(k, dl, newD):
    diff = getDistance(newD, trainD)
    prediction = tf.Variable(tf.zeros([dl, 1], dtype=tf.float32), None)
    
    for i in range(0, dl):
        row = diff[i, :]
        respon = resp(row, k)
        #Get the predicted value
        val = tf.matmul(tf.reshape(respon, [1, 80]), trainT)
        prediction = tf.scatter_add(prediction, [i], val)
    
    return prediction

def mseLoss(k, dl, newD, newT):
    
    prediction = predict(k, dl, newD)
        
    mse = tf.square(tf.subtract(newT, prediction))
    
    loss = tf.div(tf.reduce_sum(mse, 0), 2*dl)
    return loss

#training set size is 80, change dl to 80 for training set
dl = 1000

for K in [1, 3, 5, 50]:
    
    #lossTrain = mseLoss(K, 80, trainD, trainT)
    #lossValid = mseLoss(K, 10, validD, validT)
    #lossTest = mseLoss(K, 10, testD, testT)
    pNew = predict(K, dl, dataD)
    
    init_op = tf.global_variables_initializer()
    
    sess = tf.Session()
    
    sess.run([init_op])
    
    #lTr, lV, lTe = sess.run([lossTrain, lossValid, lossTest], feed_dict={dataD:X, trainD:trainData, trainT:trainTarget,\
    # validD:validData, validT:validTarget, testD:testData, testT:testTarget})

    #print "\nK is: %d \n" % K
    #print "Training: "
    #print lTr
    #print "valid: "
    #print lV
    #print "Test: "
    #print lTe
    
    #predict our new data
    pnew = sess.run(pNew, feed_dict={dataD:X, trainD:trainData, trainT:trainTarget})
    print pnew.transpose().reshape(1, 1000)




