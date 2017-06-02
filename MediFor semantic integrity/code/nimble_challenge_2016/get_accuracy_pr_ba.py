import sys
import pdb
f_name = lambda x : '../../data/nimble_data/results/pb_comp_' + str(x) + '.txt'
f_slice_name = lambda x,y : '../../data/nimble_data/results/pb_comp_' + str(x) + '_slice' + str(y) + '.txt'
# f = open('../../data/nimble_data/results/pb_comp.txt','rb')
if sys.argv[3] == 's':
    spl = True
    f = open(f_slice_name(sys.argv[1],sys.argv[2]),'rb')
else:
    spl = False
    f = open(f_name(sys.argv[1]), 'rb')

# thresh = sys.argv[2]
thresh = [0.95, 0.9, 0.8, 0.5, 0.4]
num_cor = [0 for i in range(5)]
tot_num = 0
if not spl:
    for lin in f:
        line1 = lin.split(' ')
        img_dir_num = line1[0]

        corr = float(line1[1])
    
        # corr = float(corr)
        for i in range(5):
            # pdb.set_trace()
            if (corr > thresh[i]):
                num_cor[i] += 1
        tot_num +=1
        
elif spl:
    for lin in f:
        line1 = lin.split(' ')
        img_dir_num = line1[0]

        corr1 = float(line1[1])
        corr2 = float(line1[2])
        corr = max(corr1,corr2)
        for i in range(5):
            if corr > thresh[i]:
                num_cor[i] += 1
            
        tot_num += 1
        
for i in range(5):
    print str(num_cor[i]) + '/' + str(tot_num) + ' = ' +str(num_cor[i]*1.0/tot_num)

f.close()
