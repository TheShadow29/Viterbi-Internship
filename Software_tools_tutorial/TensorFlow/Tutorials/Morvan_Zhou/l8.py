import tensorflow as tf

input1 = tf.placeholder(tf.float32,[1,2])
input2 = tf.placeholder(tf.float32,[2,1])
output1 = tf.matmul(input1,input2)

with tf.Session() as sess:
    print(sess.run(output1,feed_dict={input1 : [[3 , 3]], input2 : [[2],[2]]}))
