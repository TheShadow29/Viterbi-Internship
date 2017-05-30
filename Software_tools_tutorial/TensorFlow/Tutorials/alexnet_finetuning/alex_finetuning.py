import tensorflow as tf
class AlexNet:

    def __init__(self, x, keep_prob, num_classes, skip_layer, weights_path='default'):
        # ```
        """
        Inputs:
        - x: tf.placeholder, for the input images
        - keep_prob: tf.placeholder, for the dropout rate
        - num_classes: int, number of classes of the new dataset
        - skip_layer: list of strings, names of the layers you want to reinitialize
        - weights_path: path string, path to the pretrained weights,
        (if bvlc_alexnet.npy is not in the same folder)
        """
        # Parse input arguments
        self.X = x
        self.NUM_CLASSES = num_classes
        self.KEEP_PROB = keep_prob
        self.SKIP_LAYER = skip_layer
        self.IS_TRAINING = is_training
        
        if weights_path == 'DEFAULT':
            self.WEIGHTS_PATH = 'bvlc_alexnet.npy'
        else:
            self.WEIGHTS_PATH = weights_path

        # Call the create function to build the computational graph of AlexNet
        self.create()

    def create(self):
        
        pass

    def load_initial_weights(self):

        pass

def conv(x, filter_height, filter_width, num_filters, stride_y, stride_x, name,
         padding = 'SAME', groups = 1):

    #Get number of input channels
    input_channels = int(x.get_shape()[-1])

    #create a lambda function for convolution
    convolve = lambda i,k : tf.nn.conv2d(i,k,
                                         strides = [1, stride_y, stride_x,1],
                                         padding = padding)
    with tf.variable_scope(name) as scope:
        #create tf variables for the weights and biases of the conv layer
        weights = tf.get_variable('weights')
