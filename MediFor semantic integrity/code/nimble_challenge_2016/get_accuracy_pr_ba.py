import sys
f = open('../../data/nimble_data/pb_comp.txt','rb')

thresh = sys.argv[1]
num_cor = 0
tot_num = 0
for lin in f:
    line1 = lin.split(' ')
    img_dir_num = line1[0]
    corr = line1[1]
    if (corr > thresh):
        num_cor += 1
    tot_num += 1
print str(num_cor) + '/' + str(tot_num) + ' = ' +str(num_cor*1.0/tot_num)
