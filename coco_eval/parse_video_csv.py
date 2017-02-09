# encoding: UTF-8

import os
import glob

import numpy as np
import pandas as pd

video_src_dir = "../data/video_corpus.csv"

video_data = pd.read_csv(video_src_dir, sep=',')
video_data = video_data[video_data['Language'] == 'English']

videoID_lists = list(video_data['VideoID'])
videoID_start_lists = list(video_data['Start'])
videoID_end_lists = list(video_data['End'])
video_descriptions_lists = list(video_data['Description'])

videoID_with_Frames = []
for idx, item in enumerate(videoID_lists):
    temp = videoID_lists[idx] + '_' + str(videoID_start_lists[idx]) + '_' + str(videoID_end_lists[idx])
    videoID_with_Frames.append(temp)

#print video_descriptions_lists
