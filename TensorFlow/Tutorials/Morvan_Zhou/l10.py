import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

def add_layer(input_from_last_layer, in_size, out_size, activation_func=None):
    #create weight matrix:
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            weights = tf.Variable(tf.random_normal([in_size,out_size]),name='W')
        with tf.name_scope("bias"):
            bias = tf.Variable(tf.zeros([1,out_size]) + 0.1,name='B')
        with tf.name_scope("wx_pl_b"):
            wx_pl_b = tf.matmul(input_from_last_layer, weights) + bias

        if (activation_func == None):
            output = wx_pl_b
        else:
            output = activation_func(wx_pl_b)
        return output

x_data = np.linspace(-1,1,300)[:,np.newaxis]
noise = np.random.normal(0,0.05,x_data.shape)
y_data = np.square(x_data) - 0.5 + noise

# plt.scatter(x_data,y_data)
# plt.show()
with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None,1],name='x_input')
    ys = tf.placeholder(tf.float32, [None,1],name='y_input')

#add hidden layer
hid_layer1 = add_layer(xs,1,10,activation_func=tf.nn.relu)

#add output layer
pred = add_layer(hid_layer1,10,1,activation_func=None)

#error
with tf.name_scope("loss"):
    error = tf.reduce_mean(tf.reduce_sum(tf.square(ys - pred),reduction_indices=[1]),name='error')
with tf.name_scope("train"):    
    train = tf.train.GradientDescentOptimizer(0.1).minimize(error)

init = tf.global_variables_initializer()


#plot the data
fig = plt.figure();
ax = fig.add_subplot(1,1,1)
ax.scatter(x_data,y_data)
plt.ion()
plt.show()



with tf.Session() as sess:
    writer = tf.summary.FileWriter("logs/",sess.graph)
    sess.run(init)
    
    for i in range(1000):
        #In stochastic grad descent use a part of x_data to train the neural net
        sess.run(train, feed_dict={xs : x_data, ys : y_data})
        if i%50 == 0:
            # print (sess.run(error,feed_dict= {xs : x_data, ys : y_data}))
            try:
                ax.lines.remove(lines[0])
            except Exception:
                pass
            pred_vals = sess.run(pred, feed_dict = {xs : x_data}) #this is ind of ys
            #plot prediction
            lines = ax.plot(x_data, pred_vals,'r-',lw = 5.5)
            plt.pause(0.1)
