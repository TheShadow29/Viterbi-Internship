# import skimage
from skimage import io
# import sys
import pdb
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
fname = '/arka_data/NC2017_Dev1_Beta4/'
# l1 = sys.argv[1:]
# img = skimage.io()
io.use_plugin('matplotlib')


def show_img_protest(l1, corr_arr='none'):
    plt.figure(1)
    probe_img = io.imread(l1[0])
    plt.subplot(3, 2, 1)
    plt.imshow(probe_img)
    plt.title('Probe Image')

    for ind, f in enumerate(l1[1:]):
        img = io.imread(f)
        plt.subplot(3, 2, ind + 2)
        plt.imshow(img)
        if corr_arr == 'none':
            plt.title('World Image ' + str(ind))
        else:
            plt.title('World Image ' + str(ind) + ' ' + str(corr_arr[ind]))
    plt.show()

    
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

    
def show_all_imgs(l1):
    io.imread_collection(l1)
    io.imshow_collection(l1)

    
def show_with_bbox(img, bbox):
    """
    Expecting bbox as a 4-valued tuple
    Format <x1,y1,x2,y2>
    """
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]
    # pdb.set_trace()
    rect_height = y2 - y1
    rect_width = x2 - x1
    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    fig, ax = plt.subplots(1)
    ax.imshow(img)
    
    # Rectangle(xy, width, height, angle=0.0, **kwargs)
    # Note that in matplotlib, the lower left is (0,0)
    # Draw a rectangle with lower left at xy = (x, y) with specified width and height
    # lower left corner will be x1,y2
    rect = mpatches.Rectangle((x1, y1), rect_width, rect_height, edgecolor='r', facecolor='none')
    ax.add_patch(rect)
    plt.show()
