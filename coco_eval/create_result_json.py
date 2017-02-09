# encoding: UTF-8

import os
import glob
import numpy as np

import cPickle

#  change the output file
output_txt_dir = "../S2VT_results.txt"
output = open(output_txt_dir).read().splitlines()

num_all_output = len(output)

avi_names_lists = []
machine_produced_sentences = []

for idx, item in enumerate(output):
    if (idx % 3) == 0:
        avi_names_lists.append(item)
    if (idx % 3) == 1:
        machine_produced_sentences.append(item)

avi_npy_basenames = map(lambda item: os.path.basename(item), avi_names_lists)

avi_names = []
for each_avi in avi_npy_basenames:
    tmp1, tmp2, tmp3 = each_avi.split('.')
    avi_names.append(tmp1)

fd = open('map_videoID.pkl', 'rb')
map_videoID = cPickle.load(fd)

map_videoID_reverse = {}
for key in map_videoID:
    val = map_videoID[key]
    map_videoID_reverse[val] = key

json_fd = open('generation.json', 'w')
json_fd.write('[')
for idx, item in enumerate(avi_names):
    if idx != len(avi_names)-1:
        json_fd.write('{"image_id": ' + str(map_videoID_reverse[item]) + ', "caption": "' + str(machine_produced_sentences[idx]) + '"}, ')
    if idx == len(avi_names)-1:
        json_fd.write('{"image_id": ' + str(map_videoID_reverse[item]) + ', "caption": "' + str(machine_produced_sentences[idx]) + '"}]')

json_fd.close()
