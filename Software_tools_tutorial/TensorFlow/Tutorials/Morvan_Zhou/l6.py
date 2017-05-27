import tensorflow as tf
import numpy as np

mat1 = tf.constant([[1,3]])
mat2 = tf.constant([[2]
                    ,[2]])

prod = tf.matmul(mat1,mat2) #np.dot(m1,m2)                    

#method1
# sess = tf.Session()
# res = sess.run(prod)
# print(res)

#method2
with tf.Session() as sess:
    res2 = sess.run(prod)
    print (res2)
    
