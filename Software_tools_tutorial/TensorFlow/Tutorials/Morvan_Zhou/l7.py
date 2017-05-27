import tensorflow as tf
import numpy as np

state = tf.Variable(0,name='counter')
one = tf.constant(1)
new_val = tf.add(state, one)
update = tf.assign(state, new_val)

init = tf.global_variables_initializer()

with tf.Session() as sess:
    sess.run(init)
    for i in range(3):
        sess.run(update)
        print (sess.run(state))
