import torch
import torch.nn as nn
import torch.nn.functional as F 
import torchvision
import torchvision.transforms as transforms
import os
import pandas as pd

from torch.utils.data import Dataset
from torchvision import models
from PIL import Image
from torchvision import datasets
from torchvision.io import read_image
from torch.utils.data import DataLoader

#dir(models)
from itertools import islice
import tarfile
import pickle
from tqdm import tqdm
import numpy as np
import json


import h5py

class YouCook2DatasetLoader(Dataset):
    def __init__(self, split_type):
        self.curr_split = split_type
        splits_dir_path = '/freespace/local/YouCookII/splits' 
        if self.curr_split == 'train':
            #read train_list
            data_mapping_file_path = os.path.join(splits_dir_path, 'train_list.txt')

            with open(data_mapping_file_path) as f:
                self.train_ids = f.readlines()

        elif self.curr_split == 'val':
            data_mapping_file_path = os.path.join(splits_dir_path, 'val_list.txt')

            with open(data_mapping_file_path) as f:
                self.val_ids = f.readlines()

        elif self.curr_split == 'test':
            data_mapping_file_path = os.path.join(splits_dir_path, 'test_list.txt')
            with open(data_mapping_file_path) as f:
                self.val_ids = f.readlines()

        with open('/freespace/local/t21/Drop-DTW/data/text_features.pkl', 'rb') as handle:
            self.text_features_dataset = pickle.load(handle)

        print("Type of text_features_dataset", type(self.text_features_dataset))

        annotations_file = open('/freespace/local/YouCookII/annotations/youcookii_annotations_trainval.json')
        self.annotations = json.load(annotations_file)


    def __len__(self):
        if self.curr_split == 'train':
            return len(self.train_ids)

        elif self.curr_split == 'val':
            return len(self.val_ids)

        elif self.curr_split == 'test':
            return len(self.val_ids)

    def __getitem__(self, idx):
        video_features_dir = '/freespace/local/YouCookII/features/feat_csv/' + self.curr_split  + '_frame_feat_csv'

        data_ids = []
        if self.curr_split == 'train':
            # print("train_ids[idx] = ", self.train_ids[idx])
            video_features_dir = os.path.join(video_features_dir, self.train_ids[idx]).strip('\n')
            data_ids = self.train_ids
            
        elif self.curr_split == 'val':
            video_features_dir = os.path.join(video_features_dir, self.val_ids[idx]).strip('\n')
            data_ids = self.val_ids

        elif self.curr_split == 'test':
            video_features_dir = os.path.join(video_features_dir, self.val_ids[idx]).strip('\n')
            data_ids = self.val_idstext_features_np_list

        #read all csv files in video_features_dir
        video_features = []

        #read video features

        #take every fourth
        frame_counter = 0

        for dir in os.listdir(video_features_dir):
            dir_path = os.path.join(video_features_dir, dir)

            for f in os.listdir(dir_path):
                if f.endswith(".csv"):
                    if frame_counter % 10 == 0:
                        #print("Filename = ", f)
                        file_path = os.path.join(dir_path,f)
                        #print("File_path = ", file_path)
                        df = pd.read_csv(file_path)
                        #got the features for all segments of this video
                        #print("df shape =", df.shape)
                        
                        video_features.append(torch.from_numpy(df.values))
                    frame_counter += 1

                        
        #read text features
        video_features_tensor = torch.cat(video_features)
        print("video_features_tensor shape = ",video_features_tensor.shape)

        annotation_id = data_ids[idx].split('/')[1]
        annotation_id = annotation_id.strip("\n")
        #print('annotation_id=', annotation_id)
        current_recipe_text_features = self.text_features_dataset[annotation_id]

        
        #pack into dictionary with keys = 'frame_features', 'text_features'
        text_features_np_list = current_recipe_text_features
        text_features_np_list = text_features_np_list[0:len(text_features_np_list)//2]
        text_tensors = torch.from_numpy(text_features_np_list)

        video_text_map = {}
        #print("video_features_tensor before =", video_features_tensor.shape)

        #drop some frames
        #drop some text steps
        #Take max frames = 500
        #Take max steps = 100

        video_features_tensor = F.pad(input = video_features_tensor, pad = (0, 0, 0, 499-video_features_tensor.shape[0]), mode = "constant", value = 0)
        text_tensors = F.pad(input = text_tensors, pad = (0, 0, 0, 133-text_tensors.shape[0]), mode = "constant", value = 0)

        
        #print("video_features_tensor after =", video_features_tensor.shape)
        video_text_map['frame_features'] = video_features_tensor
        video_text_map['step_features'] = text_tensors
        return video_text_map


# yc_loader = YouCook2DatasetLoader('train')
# # # v,t = yc_loader.__getitem__(0)
# # # print(v[0].shape)
# # # print(t[0].shape)
# # # print(t.shape)

# # vt_map = yc_loader.__getitem__(0)
# # print(vt_map.keys())
# # print(vt_map['frame_features'].shape)
# # print(vt_map['step_features'].shape)


# # print(length)