import numpy as np
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = (10, 10)        # large images
plt.rcParams['image.interpolation'] = 'nearest'  # don't interpolate: show square pixels
plt.rcParams['image.cmap'] = 'gray'  # use grayscale output rather than a (potentially misleading) color heatmap

import sys
caffe_root = '../'
sys.path.insert(0, caffe_root + 'python')

import caffe
caffe.set_mode_cpu()

import numpy as np
import matplotlib.pyplot as plt

import os
if os.path.isfile(caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'):
    print 'CaffeNet found.'
else:
    print 'Downloading pre-trained CaffeNet model...'
    os.system('python ../scripts/download_model_binary.py ../models/bvlc_reference_caffenet')

labels_file = caffe_root + 'data/ilsvrc12/synset_words.txt'
if not os.path.exists(labels_file):
    os.system('sh ../data/ilsvrc12/get_ilsvrc_aux.sh')
labels = np.loadtxt(labels_file, str, delimiter='\t')

#load and run the net
model_def = caffe_root + 'models/bvlc_reference_caffenet/deploy.prototxt'
model_weights = caffe_root + 'models/bvlc_reference_caffenet/bvlc_reference_caffenet.caffemodel'

net = caffe.Net(model_def,      # defines the structure of the model
                model_weights,  # contains the trained weights
                caffe.TEST)     # for deployment (in contrast to caffe.TRAIN)

print ('Loaded Net')


# configure input pre-processing
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR
print 'Configured input.'

image = caffe.io.load_image(caffe_root + 'examples/images/cat.jpg')
transformed_image = transformer.preprocess('data', image)
# plt.imshow(transformed_image)

# reverse_transformer = caffe.io.Transformer({'data' : (227,227,3)})
# reverse_transformer.set_transpose('data',(1,2,0))
# reverse_transformer.set_channel_swap('data',(2,1,0))
# reve_img = reversed_transformer.preprocess('data',transformed_image)                             
                                    

net.blobs['data'].data[...] = transformed_image
net.forward()
output_prob = net.blobs['prob'].data[0]

print 'What does the net say?\n', labels[output_prob.argmax()]


plt.show()
