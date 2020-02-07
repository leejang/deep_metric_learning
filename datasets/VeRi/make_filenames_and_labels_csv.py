#!/usr/bin/env python
import os, glob, sys
import csv

# train
#prev_cls = '0001'
#img_path = './image_train/*.jpg'
#csv_f_path = './filenames_w_labels.csv'

# query
#prev_cls = '0002'
#img_path = './image_query/*.jpg'
#csv_f_path = './filenames_w_labels_query.csv'

# test
prev_cls = '0002'
img_path = './image_test/*.jpg'
csv_f_path = './filenames_w_labels_test.csv'

# file open
csv_f = open(csv_f_path, 'wt')
# csv writer
writer = csv.writer(csv_f)

new_label = 0

for img in sorted(glob.glob(img_path)):

    fname = os.path.basename(img)
    cls = fname[:4]

    if (cls != prev_cls):
        new_label += 1

    print (fname, cls, new_label)
    #fname = 'image_train/' + fname
    #fname = 'image_query/' + fname
    fname = 'image_test/' + fname
    writer.writerow([new_label,fname])

    prev_cls = cls


csv_f.close()
