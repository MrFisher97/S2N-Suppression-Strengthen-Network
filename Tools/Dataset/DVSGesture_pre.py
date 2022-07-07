"""
Download DVS128 Gesture Dataset from:http://research.ibm.com/dvsgesture/
This code is used to extract Datasets (in .aedat format) into .h5
AedatTool is required.
Download AedatTool from: https://github.com/qiaokaki/AedatTools
.h file format:
'data': N x seq_len x num_events x 3 
'label': N x seq_len
Author: Wang Qinyi
Date: Jan 2018

Modified: Fisher
Date: April 2021
"""

import sys
import os
import numpy as np
import pandas as pd
import re
import glob
import shutil
import concurrent.futures
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
from dv import AedatFile
import Tools.Dataset.generator as generator

np.random.seed(2021)
DATA_PATH = '/home/chjz/workspace/wzy/DVS/Dataset/DVSGesture' # the place of database
SAVE_PATH = '/home/chjz/workspace/wzy/DVS/Dataset/DVSGesture_stream' # the place you want to stor your data
S2US = 1e6

def filter_file(read_file, store_file, cond):
    context = ''
    with open(read_file, 'r') as f:
        for row in f.readlines():
            if re.match(cond, row):
                context += row
    print(context)
    with open(store_file, 'a') as f:
        f.write(context)

def get_file_list(mode = 'train', scene = 'all'):
    if scene == 'all':
        if mode == 'train':
            data_files = glob.glob(os.path.join(DATA_PATH, 'user[0-1]*.aedat4'))
            data_files += glob.glob(os.path.join(DATA_PATH, 'user2[0-4]*.aedat4'))
        else:
            data_files = glob.glob(os.path.join(DATA_PATH, 'user2[5-9]*.aedat4'))
    else:
        if mode == 'train':
            data_files = glob.glob(os.path.join(DATA_PATH, f'user[0-1]*{scene}.aedat4'))
            data_files += glob.glob(os.path.join(DATA_PATH, f'user2[0-4]*{scene}.aedat4'))
        else:
            data_files = glob.glob(os.path.join(DATA_PATH, f'user2[5-9]*{scene}.aedat4'))
    return data_files

def get_window_index(timestamps, stepsize, windowsize):
    """
    Extract each class from original video
    """
    win_start_index = []
    win_end_index = []
    idx = 0
    start_wins = [timestamps[0] + i * stepsize for i in range(int((timestamps[-1] - timestamps[0] - windowsize) // stepsize) + 1)]
    end_wins = [sw + windowsize for sw in start_wins]
    for i in range(len(start_wins)):
        win_start_index.append((np.argwhere(timestamps >= start_wins[i])).min())
        win_end_index.append((np.argwhere(timestamps >= end_wins[i])).min())
    return win_start_index, win_end_index

def generate_cnt_frames(win_size, events):
    nframe = 15
    frames = np.zeros([2, nframe, 128, 128], dtype=np.uint8) #C * F * H * W
    win_st, win_ed = get_window_index(events[:, 0], win_size / nframe, win_size / nframe)
    for f in range(len(win_st)):
        win_events = events[win_st[f]:win_ed[f]]
        for i in range(len(win_events)):
            if win_events[i, 3] == 1:
                # calculate the counts for positive events
                frames[0, f, (int)(win_events[i, 1]), (int)(win_events[i, 2])] = (
                    frames[0, f, (int)(win_events[i, 1]), (int)(win_events[i, 2])] + 1
                )
            else:
                # calculate the counts for negtive events
                frames[1, f, (int)(win_events[i, 1]), (int)(win_events[i, 2])] = (
                    frames[1, f, (int)(win_events[i, 1]), (int)(win_events[i, 2])] + 1
                )
    return frames

def process_file(name, export_path, nclass, clip_size, fps):
    data, label = [], []
    print('Processing Data File: ', name)
    # time_label[1] = time_label[1].astype(float)
    label_file = name.split('/')[-1].split('.')[0] + '_labels.csv'
    user_scene = name.split('/')[-1].split('.')[0]
    time_label = pd.read_csv(os.path.join(DATA_PATH, label_file), delimiter = ',')
    dt = 1 / fps
    nframe = int(fps * clip_size)
    #-------------Extract raw data (timestep,x,y)----------------------- 
    
    with AedatFile(name) as f:
        # Access dimensions of the event stream
        height, width = f['events'].size
        events = np.hstack([packet for packet in f['events'].numpy()])
        # Access information of all events by type
        t, x, y, p = events['timestamp'], events['x'], events['y'], events['polarity']
        t -= min(t)
        t = t + time_label['startTime_usec'][0]
        t = t / 1e6

        events = np.vstack((t, x, y, p)).transpose(1, 0)
        #---------Extract data by sliding window for each class and generate the point cloud data--------------
        for i in range(len(time_label['class'])):
            if time_label['class'][i] > nclass:
                continue
            
            st = time_label['startTime_usec'][i] / 1e6
            ed = time_label['endTime_usec'][i] / 1e6

            label = time_label['class'][i]
            nclip = int((ed - st) / clip_size)
            clip_ts = np.arange(st, ed, clip_size)
            index_array = np.searchsorted(t, clip_ts)
            for n in range(nclip):
                clip_events = events[index_array[n]:index_array[n+1]]
                clip_events[:, 0] -= min(clip_events[:, 0])
                np.save(os.path.join(export_path, f'{label}_{user_scene}_N{n}.npy'), clip_events.astype(np.float16))
    print('Processing Data File: ', name, 'Done !')
    return name

def generate_dataset(nclass, clip_size, fps, scene = 'natural'):
    '''
    nclass: number of action class (<=8)
    win_size: size of time window (e.g 5 is 5s)
    step_size: the size between two windows (e.g 1 is 1s)
    seq_len: number of time window used as one valid sample
    npoint: number of points sampled in a time window
    database: which type of data['DvsGesture', 'DVSAction']
    scene: the type of scene ['fluorescent', 'fluorescent_led', 'natural', 'all'], 'all' means all scenes

    output .h5 data file in SAVE_PATH(defined at front)
    '''

    # generate the output path of data file
    foldername = 'C' + str(nclass) + '_W' + str(clip_size).replace('.','')
    if scene == '':
        scene = 'all'
    foldername += '_' + scene
    export_path = os.path.join(SAVE_PATH, foldername)
    if os.path.exists(export_path):
        shutil.rmtree(export_path)
    
    for mode in ['train', 'test']:
        path = os.path.join(export_path, mode)
        os.makedirs(path)
        print('Data will save to', path)
        data_files = get_file_list(mode, scene)
        
        # single thread
        # for name in data_files:
        #     process_file(name, path, nclass,  clip_size, fps)
        
        # multiple thread
        with concurrent.futures.ProcessPoolExecutor(max_workers=8) as executor:
            tasks = []
            for name in data_files:
                tasks.append(executor.submit(process_file, name, path, nclass, clip_size, fps))
            concurrent.futures.wait(tasks)
            print(f'DVSGesture dataset at {scene} created')

def creat_h5(root):
    import h5py
    with h5py.File('C11_W05.h5', 'w') as h5f:
        fileList = glob.glob(os.path.join(root, 'train', '*'))
        trainGroup = h5f.create_group("train")
        for f in fileList:
            data = np.load(f)
            fName = f.split('/')[-1].split('.')[0]
            trainGroup.create_dataset(fName, data=data)

        fileList = glob.glob(os.path.join(root, 'test', '*'))        
        testGroup = h5f.create_group("test")
        for f in fileList:
            data = np.load(f)
            fName = f.split('/')[-1].split('.')[0]
            testGroup.create_dataset(fName, data=data)

if __name__ == "__main__":
    # from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
    # root_dir = '/home/chjz/workspace/wzy/DVS/Dataset/DVSGesture'
    # events_np_root = '/home/chjz/workspace/wzy/DVS/Dataset/DVSGesture/events_np'
    # train_set = DVS128Gesture.create_events_np_files(root_dir, events_np_root)
    # creat_h5('/home/chjz/workspace/wzy/DVS/Dataset/DVSGesture/C11_W05')
    # for scene in ['all']:
    # # for scene in ['fluorescent', 'led', 'natural', ]:
    #     generate_dataset(10, 0.5, 30, scene=scene)
    pass