import two_imgs
import os
import pdb
if __name__ == '__main__':
    
    img_top_dir = '../../data/nimble_data/manipulated/'
    res = ''
    i = 0
    for subdir in os.listdir(img_top_dir):
    # subdir = '601'
        bo = True
        img_dir = img_top_dir + subdir + '/'
        dir_paths = os.listdir(img_dir)
        imgs_paths = []
        for k in dir_paths:
            if k[-4:] == '.jpg' or k[-4:] == '.png':
                imgs_paths.append(k)
        if len(imgs_paths) == 2:
            img1_path = img_dir + imgs_paths[0]
            img2_path = img_dir + imgs_paths[1]
            try:
                to_print = two_imgs.are_the_two_imgs_same(img1_path, img2_path)
            except Exception as e :
                bo = False
                pass
            # pdb.set_trace()
            # with open(img_dir + 'eval.txt', 'w') as f:
            if bo == True:
                f = open(img_dir + 'eval.txt','w')
                f.write(str(to_print))
                # f.write('acb')
                f.close()
                res += str(subdir) + str(to_print[3]['pear_ncc']) + '\n'
            i += 1
            print ('Iter: ' + str(i) +  ' Counter ' + str(subdir))
    g = open('../../data/nimble_data/pb_comp.txt','w')
    g.write(res)
    g.close()
