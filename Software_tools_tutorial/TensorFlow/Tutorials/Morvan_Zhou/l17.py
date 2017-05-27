import tensorflow as tf
from sklearn.datasets import load_digits
from sklearn.cross_validation import train_test_split
from sklearn.preprocessing import LabelBinarizer

#load data
digits = load_digits()
X = digits.data
y = digits.target
y = LabelBinarizer().fit_transform(y)
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3)

def add_layer(inputs, in_size, out_size, layer_name, activation_function=None):
    # add one more layer and return the output of this layer
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1, )
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    # here to dropout
    # Wx_plus_b = tf.nn.dropout(Wx_plus_b, keep_prob)
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b, )
    tf.summary.histogram(layer_name + '/outputs', outputs)
    return outputs

#place holder
xs = tf.placeholder(tf.float32, [None,64]) #8x8
ys = tf.placeholder(tf.float32, [None, 10])

#layer1
l1 = add_layer(xs, 64, 50, 'l1', activation_function=tf.nn.tanh)
pred = add_layer(l1, 50, 10, 'l2', activation_function = tf.nn.softmax) 

#cross entropy
cross_entropy = tf.reduce_mean(-tf.reduce_sum(ys*tf.log(pred),
                                              reduction_indices = [1]))
tf.summary.scalar('loss',cross_entropy)
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

sess = tf.Session()
merged = tf.summary.merge_all()

train_writer = tf.summary.FileWriter("logs17/train", sess.graph)
test_writer = tf.summary.FileWriter("logs17/test", sess.graph)

sess.run(tf.global_variables_initializer())


for i in range(500):
    sess.run(train_step, feed_dict={xs:X_train, ys:y_train})

    if i%50 == 0:
        train_result = sess.run(merged, feed_dict={xs: X_train, ys: y_train}) 
        test_result = sess.run(merged, feed_dict={xs: X_test, ys: y_test})
        train_writer.add_summary(train_result, i)
        test_writer.add_summary(test_result, i)
