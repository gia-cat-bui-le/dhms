import glob
import os
import pickle
import shutil
from pathlib import Path
import numpy as np
import torch
import sys
import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
sys.path.append(os.getcwd()) 

def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

def get_train_test_list(dataset_path):
    filter_list = set(fileToList(f"{dataset_path}/splits/ignore_list.txt"))
    train_list = set(fileToList(f"{dataset_path}/splits/crossmodal_train.txt"))
    test_list = set(fileToList(f"{dataset_path}/splits/crossmodal_test.txt"))
    val_list = set(fileToList(f"{dataset_path}/splits/crossmodal_val.txt"))
    test_list.update(val_list)
    
    return filter_list, train_list, test_list

def get_train_test_list_finedance(dataset_path):
    
    data_split = {}
    all_list = []
    train_list = []
    for i in range(1,212):
        all_list.append(str(i).zfill(3))
    test_list = ["001","002","003","004","005","006","007","008","009","010","011","012","013","124","126","128","130","132"]
    val_list = ["115","117","119","121","122","135","137","139","141","143","145","147"]
    for one in all_list:
        if one not in test_list:
            if one not in val_list:
                train_list.append(one)

    data_split["train"] = train_list
    data_split["test"] = test_list
    data_split["val"] = val_list
    data_split["ignore"] =  ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
    
    return train_list, test_list, val_list

def split_data(dataset_path):
    filter_list, train_list, test_list = get_train_test_list(dataset_path)
    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{dataset_path}/{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"{dataset_path}/{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            if sequence in filter_list:
                continue
            motion = f"{dataset_path}/motions/{sequence}.pkl"
            wav = f"{dataset_path}/wavs/{sequence}.wav"
            assert os.path.isfile(motion)
            assert os.path.isfile(wav)
            motion_data = pickle.load(open(motion, "rb"))
            trans = motion_data["smpl_trans"]
            pose = motion_data["smpl_poses"]
            scale = motion_data["smpl_scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"{dataset_path}/{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"{dataset_path}/{split_name}/wavs/{sequence}.wav")
            
def split_data_finedance(dataset_path):
    train_list, test_list, val_list = get_train_test_list_finedance(dataset_path)
    # train - test split
    for split_list, split_name in zip([train_list, test_list], ["train", "test"]):
        Path(f"{dataset_path}/{split_name}/motions").mkdir(parents=True, exist_ok=True)
        Path(f"{dataset_path}/{split_name}/wavs").mkdir(parents=True, exist_ok=True)
        for sequence in split_list:
            motion = f"{dataset_path}/motion_fea319/{sequence}.pkl"
            wav = f"{dataset_path}/music_npy/{sequence}.wav"
            assert os.path.isfile(motion)
            assert os.path.isfile(wav)
            motion_data = pickle.load(open(motion, "rb"))
            trans = motion_data["smpl_trans"]
            pose = motion_data["smpl_poses"]
            scale = motion_data["smpl_scaling"]
            out_data = {"pos": trans, "q": pose, "scale": scale}
            pickle.dump(out_data, open(f"{dataset_path}/{split_name}/motions/{sequence}.pkl", "wb"))
            shutil.copyfile(wav, f"{dataset_path}/{split_name}/wavs/{sequence}.wav")