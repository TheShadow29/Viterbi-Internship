import tensorflow as tf
import numpy as np

#create data
x_data = np.random.rand(100).astype(np.float32)
y_data = x_data*0.1 + 0.3

#TF structures
weights = tf.Variable(tf.random_uniform([1],-1.0,1.0))
bias = tf.Variable(tf.zeros([1]))

y_predicted = weights * x_data + bias
error = tf.reduce_mean(tf.square(y_predicted - y_data))
opt_grd = tf.train.GradientDescentOptimizer(0.5)
train = opt_grd.minimize(error)
# init = tf.initialize_all_variables()
init = tf.global_variables_initializer()
#TF session
sess = tf.Session()
sess.run(init)

for i in range(201):
    sess.run(train)
    if  i%20 == 0:
        print (i, sess.run(weights), sess.run(bias))
