# import skimage
from skimage import io
# import sys
# import pdb
import matplotlib.pyplot as plt

fname = '/arka_data/NC2017_Dev1_Beta4/'
# l1 = sys.argv[1:]
# img = skimage.io()
io.use_plugin('matplotlib')


def show_img(l1):
    plt.figure(1)
    # plt.subplot()
    full_file_name = []
    for i in range(len(l1)):
        full_file_name.append(fname + l1[i])
    f = full_file_name[0]
    # pdb.set_trace()
    probe_img = io.imread(f + '.jpg')
    plt.subplot(3, 2, 1)
    plt.imshow(probe_img)
    plt.title('Probe Image')
    for ind, f in enumerate(full_file_name[1:]):
        # pdb.set_trace()
        img = io.imread(f)
        plt.subplot(3, 2, ind+2)
        plt.imshow(img)
        plt.title('world_img ' + str(ind))

    plt.show()
