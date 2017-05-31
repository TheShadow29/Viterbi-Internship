import caffe
import numpy as np
import matplotlib.pyplot as plt
import pdb
proto_top_dir = '/home/arka_s/Caffe/caffe/models/bvlc_googlenet/'
labels_file = '/home/arka_s/Caffe/caffe/data/ilsvrc12/synset_words.txt'
net = caffe.Net(proto_top_dir + 'deploy.prototxt', proto_top_dir + 'bvlc_googlenet.caffemodel', caffe.TEST)



transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_mean('data', np.array([104,117,123]))

transformer.set_transpose('data',(2,0,1))
transformer.set_channel_swap('data',(2,1,0))
transformer.set_raw_scale('data', 255.0)
    

image_path = 'cat.jpg'

img = caffe.io.load_image(image_path)

img = caffe.io.resize( img, (224, 224, 3) )
# plt.imshow(img)
# plt.show()

# 224,224,3 -> 3,224,224
# img = np.transpose(img, (2, 0, 1))
# img = caffe.
# mean subtraction
# img[0,:,:] -= 104 / 255.0
# img[1,:,:] -= 117 / 255.0
# img[2,:,:] -= 123 / 255.0

# img = img * 255


# net.blobs['data'].reshape(1,3,224,224)
net.blobs['data'].reshape(1,3,224,224)
net.blobs['data'].data[:,:,:] = transformer.preprocess('data',img)

# out = net.forward(data=np.array([img]))['prob']
output = net.forward()
out = net.blobs['prob'].data[0].flatten()
labels = np.loadtxt(labels_file, str, delimiter='\t')
print(np.argmax(out))
print ('output label : ' + labels[out.argmax()])
