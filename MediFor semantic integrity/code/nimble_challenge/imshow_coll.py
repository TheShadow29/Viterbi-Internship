from skimage import io
import os
import matplotlib.pyplot as plt


def imshow_collection_new(l1):
    if type(l1) == str:
        file_list = os.listdir(l1)
        file_list = [l1 + f if l1[-1] == '/' else l1 + '/' + f for f in file_list]
    elif type(l1) == list:
        file_list = l1
    n = len(file_list)
    nrow = int(n**0.5)
    if n % nrow == 0:
        ncol = n/nrow
    else:
        ncol = n/nrow + 1
    plt.figure()
    for i, f in enumerate(file_list):
        plt.subplot(nrow, ncol, i+1)
        plt.imshow(io.imread(f))
    plt.show()
    return plt
