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

from finedance.render_joints.smplfk import SMPLX_Skeleton, do_smplxfk, ax_to_6v, ax_from_6v

def fileToList(f):
    out = open(f, "r").readlines()
    out = [x.strip() for x in out]
    out = [x for x in out if len(x)]
    return out

def finedance_train_test_split():
    all_list = []
    train_list = []
    for i in range(1,212):
        all_list.append(str(i).zfill(3))

    test_list = ["063", "132", "143", "036", "098", "198", "130", "012", "211", "193", "179", "065", "137", "161", "092", "120", "037", "109", "204", "144"]
    ignor_list = ["116", "117", "118", "119", "120", "121", "122", "123", "202"]
    tradition_list = ['005', '007', '008', '015', '017', '018', '021', '022', '023', '024', '025', '026', '027', '028', '029', '030', '032', '032', '033', '034', '035', '036', '037', '038', '039', '040', '041', '042', '043', '044', '045', '046', '047', '048', '049', '050', '051', '072', '073', '074', '075', '076', '077', '078', '079', '080', '081', '082', '083', '104', '105', '106', '107', '108', '109', '110', '111', '112', '113', '114', '115', '116', '117', '118', '119', '120', '121', '122', '123', '126', '127', '132', '133', '134',  '135', '136', '137', '138', '139', '140', '141', '142', '143', '144', '145', '146', '147', '148', '151', '152', '153', '154', '155', '170']
    morden_list = []
    for one in all_list:
        if one not in tradition_list:
            morden_list.append(one)

    ignor_list = ignor_list
    for one in all_list:
        if one not in test_list:
            train_list.append(one)

    return ignor_list, train_list, test_list

def get_train_test_list(dataset_path, dataset_name):
    if dataset_name == "aistpp":
        filter_list = set(fileToList(f"{dataset_path}/splits/ignore_list.txt"))
        train_list = set(fileToList(f"{dataset_path}/splits/crossmodal_train.txt"))
        test_list = set(fileToList(f"{dataset_path}/splits/crossmodal_test.txt"))
    elif dataset_name == "finedance":
        filter_list, train_list, test_list = finedance_train_test_split()
    return filter_list, train_list, test_list

def split_data(dataset_path, dataset_name):
    filter_list, train_list, test_list = get_train_test_list(dataset_path, dataset_name)
    # train - test split
    if dataset_name == "aistpp":
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