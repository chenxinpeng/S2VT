# encoding: UTF-8

import os
import glob
import numpy as np
import pandas as pd
import cPickle as pickle


video_src_dir = "../data/video_corpus.csv"

video_data = pd.read_csv(video_src_dir, sep=',')
video_data = video_data[video_data['Language'] == 'English']

videoID_lists = list(video_data['VideoID'])
videoID_start_lists = list(video_data['Start'])
videoID_end_lists = list(video_data['End'])
video_descriptions_lists = list(video_data['Description'])

videoID_with_Frames = []
for idx, item in enumerate(videoID_lists):
    temp = videoID_lists[idx] + '_' + str(int(videoID_start_lists[idx])) + '_' + str(int(videoID_end_lists[idx]))
    videoID_with_Frames.append(temp)

# videoID map
videoID_shrinked = list(set(videoID_with_Frames))

map_videoID = {}
for idx, item in enumerate(videoID_shrinked):
    map_videoID[idx] = item

with open('map_videoID.pkl', 'w') as f:
    pickle.dump(map_videoID, f)

###########################################################################################################
# judge the ascii
###########################################################################################################
def is_ascii(s):
    return all(ord(c) < 128 for c in s)

json_fd = open('reference.json', 'w')
json_fd.write('{"info": {"description": "test", "url": "https://github.com/chenxinpeng", "version": "1.0", "year": 2017, "contributor": "ChenXinpeng", "date_created": "2017-01-27"}, "images": [')
for key in map_videoID:
    json_fd.write('{"license": 1, "file_name": "' + str(map_videoID[key]) + '", "id": ' + str(key) + '}, ')

json_fd.write('], "licenses": [{"url": "http://creativecommons.org/licenses/by-nc-sa/2.0/", "id": 1, "name": "Attribution-NonCommercial-ShareAlike License"}], ')

json_fd.write('"type": "captions", "annotations": [')

id_count = 0
for key in map_videoID:
    video_frame = map_videoID[key]
    indices = [i for i, x in enumerate(videoID_with_Frames) if x == video_frame]
    for idx in indices:
        if type(video_descriptions_lists[idx]) == type(1.0):
            continue
        
        elif '\\' in video_descriptions_lists[idx]:
            print video_descriptions_lists[idx]
            continue
        
        #elif '/' in video_descriptions_lists[idx]:
        #    print video_descriptions_lists[idx]
        #    continue
        elif '"' in video_descriptions_lists[idx]:
            print video_descriptions_lists[int(idx)]
            continue
        
        elif "\n" in video_descriptions_lists[idx]:
            print video_descriptions_lists[int(idx)]
            continue
        
        #elif "'" in video_descriptions_lists[idx]:
        #    print video_descriptions_lists[idx]
        #    continue
        #elif ":" in video_descriptions_lists[idx]:
        #    print video_descriptions_lists[idx]
        #    continue
        
        elif is_ascii(video_descriptions_lists[idx]):
            json_fd.write('{"image_id": ' + str(key) + ', "id": ' + str(id_count) + ', "caption": "' + str(video_descriptions_lists[idx]) + '"}, ')
            id_count = id_count + 1


json_fd.write("]}")
json_fd.close()
