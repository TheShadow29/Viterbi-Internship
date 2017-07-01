from __future__ import division
import os
import sys
import tensorflow as tf
import skimage.io as io
import numpy as np

sys.path.append('/home/arka_s/internship_files/image_segmentation/tf-image-segmentation/')
sys.path.append('/home/arka_s/internship_files/image_segmentation/models/slim/')

fcn_16s_checkpoint_path = '/home/arka_s/internship_files/image_segmentation/fcn_8s_checkpoint/model_fcn8s_final.ckpt'
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
slim = tf.contrib.slim

from tf_image_segmentation.models.fcn_8s import FCN_8s
from tf_image_segmentation.utils.inference import adapt_network_for_any_size_input
from tf_image_segmentation.utils.pascal_voc import pascal_segmentation_lut

number_of_classes = 21
img_fname = '../../data/nimble17_data/cluster_folder/0/1657074819a8525b50a1c375716d252f.png'
img_fname_placeholder = tf.placeholder(tf.string)
feed_dict_to_use = {img_fname_placeholder: img_fname}
img_tensor = tf.read_file(img_fname_placeholder)

image_tensor = tf.read_file(img_fname_placeholder)
if img_fname[-4:] == '.jpg':
    image_tensor = tf.image.decode_jpeg(image_tensor, channels=3)
elif img_fname[-4:] == '.png':
    image_tensor = tf.image.decode_png(image_tensor, channels=3)
