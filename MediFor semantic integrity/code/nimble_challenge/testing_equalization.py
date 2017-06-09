# coding: utf-8
img1_path
img1_path = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/beach_mdf_019.jpg'
img2_path = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Pruned_Protest_YFCCImages/protest_img_019.jpg'
from skimage import io
get_ipython().magic(u'matplotlib')
io.imshow(img1_path); io.imshow(img2_path)
import matplotlib.pyplot as plt
plt.figure(); io.imshow(img1_path); plt.figure; io.imshow(img2_path)
plt.figure(); io.imshow(img1_path); plt.figure(); io.imshow(img2_path)
get_ipython().magic(u'run two_imgs_eff.py')
are_the_two_imgs_same(img1_path, img2_path)
f = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/mod_fine_labels.npy'; a1 = np.load(f); a1 = a1.item()
a1
img1_path = '/home/nkovvuri/Rama_Work/dataset/Protest_Images/Modified_Images_ProtestL/beach_mdf_019.jpg'
bbx = a1['beach_mdf_019']
bbx
io.imshow(img1_path); io.imshow(img2_path)
plt.figure(); io.imshow(img1_path); plt.figure(); io.imshow(img2_path)
img1 = io.imread(img1_path)
img2 = io.imread(img2_path)
plt.figure(); io.imshow(img1); plt.figure(); io.imshow(img2)
img11 = img1[bbx[1]:bbx[3], bbx[0]:bbx[2]]
plt.figure(); io.imshow(img11); plt.figure(); io.imshow(img2)
help(io.imsave)
get_ipython().magic(u'mkdir ../../data/protest_data/cropped')
io.imsave('../../data/protest_data/cropped/beach_cropped_19')
io.imsave('../../data/protest_data/cropped/beach_cropped_19', img11)
io.imsave('../../data/protest_data/cropped/beach_cropped_19.jpg', img11)
io.imshow('../../data/protest_data/cropped/beach_cropped_19.jpg')
img11_path = '../../data/protest_data/cropped/beach_cropped_19.jpg'
are_the_two_imgs_same(img11_path, img2_path)
img12 = skimage.exposure.equalize_hist(img11)
rgb2gray
skimage.rgb2gray
img12 = img11
img12[:,:,i] = skimage.exposure.equalize_hist(img11[:,:,i])
for i in range(3):
    img12[:,:,i] = skimage.exposure.equalize_hist(img11[:,:,i])
    
io.imshow(img12)
img12.shape
img11.shape
io.imshow(img11)
img11 = io.imread('../../data/protest_data/cropped/beach_cropped_19.jpg')
io.imshow(img11)
for i in range(3):
    img12[:,:,i] = skimage.exposure.equalize_hist(img11[:,:,i])
    
io.imshow(img12)
io.imshow(img11)
img11
img12
img11
img11.dtype = float64
img11.dtype = float32
img13 = float(img11)
io.imshow(img11[:,:,0])
io.imshow(img12[:,:,0])
img12
skimage.exposure.histogram(img11)
plt.hist(img11)
plt.hist(img11[:,:,0])
plt.hist(img11[:,:,0])
plt.hist(img11[:,:,0].flatten())
help(plt.hist)
plt.hist(img11[:,:,0].flatten(),50)
plt.hist(img11[:,:,0].flatten(),100)
img2[:,:,0] = skimage.exposure.equalize_hist(img11[:,:,0])
img12[:,:,0] = skimage.exposure.equalize_hist(img11[:,:,0])
plt.hist(img12[:,:,0].flatten())
img12.dtype
img13 = skimage.exposure.equalize_hist(img11[:,:,0])
plt.hist(img13.flatten())
io.imshow(img13)
type(img1_path)
type(img1_path) == 'str'
type(img1_path) == str
get_ipython().magic(u'run two_imgs_eff.py')
are_the_two_imgs_same(img11,img12)
io.imshow(img11)
are_the_two_imgs_same(img11,img13)
img13 = np.array(np.zeros(img11.shape))
img13.shape
img11.shape
img13 = skimage.exposure.equalize_hist(img11)
io.imshow(img13)
io.imshow(img11)
plt.figure(); io.imshow(img11); plt.figure(); io.imshow(img13)
img13
img14 = img13
img13.astype(int)
img15 = img14 * 255
img15
img15.astype(int)
img14
img15
img14 = img15.astype(int)
img14
io.imshow(img14)
img15
img15.astype(unsigned_short)
io.imshow(img14)
img11.dtype
img15.astype(uint8)
img15.astype('uint8')
img14 = img15.astype(uint8)
img14 = img15.astype('uint8')
io.imshow(img14)
plt.hist(img14)
plt.hist(img14.flatten())
img2
io.imshow(img2)
img21 = np.array(np.zeros(img2.shape))
img21 = skimage.exposure.equalize_hist(img2)
io.imshow(img21)
plt.figure(); io.imshow(img2); plt.figure(); io.imshow(img21)
plt.figure(); io.imshow(img1); plt.figure(); io.imshow(img14)
plt.figure(); io.imshow(img11); plt.figure(); io.imshow(img14)
plt.close('all')
