import os

# tdir = '../../data/protest_data/cropped/direct_cropped/'
tdir = '../../data/protest_data/cropped/hist_equalized/'
for f in os.listdir(tdir):
    if f[-4:] == '.jpg' or f[-4:] == '.png':
        new_fname = f.split('.')[0].split('_')
        if not new_fname[-1].isdigit() and new_fname[-1] == 'cropped':
            new_fname[-1], new_fname[-2] = new_fname[-2], new_fname[-1]
            new_file_name = '_'.join(new_fname) + f[-4:]
            src = tdir + f
            dest = tdir + new_file_name
            os.rename(src, dest)
        elif not new_fname[-1].isdigit() and new_fname[-1] == 'heq':
            new_fname[-1], new_fname[-2], new_fname[-3] = new_fname[-3], new_fname[-1], new_fname[-2]
            new_file_name = '_'.join(new_fname) + f[-4:]
            src = tdir + f
            dest = tdir + new_file_name
            os.rename(src, dest)
