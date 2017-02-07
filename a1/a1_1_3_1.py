import tensorflow as tf

def findResp(distance, k):   

    #Use the negative of the distance, so later when you find the maximum, it is actually the minimum
    nd = tf.mul(distance, -1)
    
    values, indices = tf.nn.top_k(nd, k, False, None)
    
    base = tf.Variable(tf.zeros([6], dtype=tf.float32), None)
    
    update = tf.ones(k, tf.float32)
    
    res_one = tf.scatter_add(base, indices, update, None, None)
    
    res = tf.div(res_one, k)
    
    return res

k = 3

distance = tf.constant([1, 2, 3, 4, 5, 6]) 

res = findResp(distance, k)

init_op = tf.global_variables_initializer()

sess = tf.Session()

sess.run([init_op])

print sess.run(res)
    
