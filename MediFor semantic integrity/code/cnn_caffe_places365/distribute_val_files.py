
f = open('/arka_data/places_data/filelist_places365-standard/places365_val.txt','rb')

# lines = [line.rstrip('\n') for line in f]
lines = f.readlines()

total_lines = len(lines)
txt_file_name_td = '/arka_data/places_data/filelist_places365-standard/places365_val'
count = 0
step = total_lines/10
for i in range(0,total_lines,step):
    count += 1 
    f1 = open(txt_file_name_td + '_' + str(count) + '.txt','w')
    for line in lines[i:i+step]:
        f1.write(line)
    f1.close()

