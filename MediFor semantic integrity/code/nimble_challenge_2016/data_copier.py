import os
import sys
import parse_all_data
import shutil

man_ref_file = parse_all_data.man_ref_file

man_nimble_ref = parse_all_data.nimble_references(man_ref_file)
man_nimble_ref.populate_data()

# img_path_orig = lambda x,y : '/arka_data/NC2016_Test0613/' + str(x) + '/NC2016_' + '%04d' %y + '.jpg'
img_path_l = lambda x : '/arka_data/NC2016_Test0613/' + str(x)
dest_path_l = lambda x : '../../data/nimble_data/manipulated/' + str(x)
# dest_path_m = lambda x,y : dest_path_l(x) + '/' + str(y) 
# dest_path_m = lambda x : os.mkdir(dest_path_l(x)) if not os.path.isdir(dest_path_l(x))
counter = 0
for man_ref in man_nimble_ref.data:
    counter += 1
    if man_ref.collection == 'Nimble-WEB' and man_ref.is_control == 'N' and man_ref.probe_mask_file_name != '':
        if not os.path.isdir(dest_path_l(counter)):
            os.mkdir(dest_path_l(counter))
        try:
            shutil.copy2(img_path_l(man_ref.probe_file_name),dest_path_l(counter))
            shutil.copy2(img_path_l(man_ref.base_file_name), dest_path_l(counter))
        except Exception as e:
            print type(e)(' err at %s ' %counter)
            pass
