#!/usr/bin/env python
import os, glob, sys
import csv

# query
img_path = './image_query/*.jpg'
csv_f_path = './filenames_w_pids_query.csv'

# test
#img_path = './image_test/*.jpg'
#csv_f_path = './filenames_w_pids_test.csv'

# file open
csv_f = open(csv_f_path, 'wt')
# csv writer
writer = csv.writer(csv_f)

for img in sorted(glob.glob(img_path)):

    fname = os.path.basename(img)
    cls = fname[:4]

    print (fname, cls)
    #fname = 'image_train/' + fname
    fname = 'image_query/' + fname
    #fname = 'image_test/' + fname
    writer.writerow([cls,fname])

csv_f.close()
