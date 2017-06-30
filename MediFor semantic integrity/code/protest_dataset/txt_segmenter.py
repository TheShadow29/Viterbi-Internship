import skimage as ski
import skimage.io as skio
import os


img1_tdir = '../../data/protest_data/only_text/beach_mdf_019/'
img1_list = os.listdir(img1_tdir)
img1 = skio.imread(img1_tdir + img1_list[1])
skio.imshow(img1)
skio.show()
