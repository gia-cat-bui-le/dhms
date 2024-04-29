import argparse
import os
from pathlib import Path

from audio_extraction.baseline_features import \
    extract_folder as baseline_extract
from audio_extraction.jukebox_features import extract_folder as jukebox_extract
from filter_split_data import *
from slice import *


def create_dataset(opt):
    path_folder = ''
    if opt.dataset_name == "aistpp":
        path_folder = os.path.join(opt.datapath, 'aistpp_dataset/')
    elif opt.dataset_name == 'finedance':
        path_folder = os.path.join(opt.datapath, 'finedance/')
    # # split the data according to the splits files
    print("Creating train / test split")
    split_data(path_folder, opt.dataset_name)
    
    # process dataset to extract audio features
    if opt.dataset_name == "aistpp":
        if opt.extract_baseline:
            extract_feature = 'baseline'
            print("Extracting baseline features")
            baseline_extract(f"{path_folder}/train/wavs", f"{path_folder}/train/music_npy")
            baseline_extract(f"{path_folder}/test/wavs", f"{path_folder}/test/music_npy")
        if opt.extract_jukebox:
            extract_feature = 'jukebox'
            print("Extracting jukebox features")
            jukebox_extract(f"{path_folder}/train/wavs_sliced", f"{path_folder}/train/jukebox_feats")
            jukebox_extract(f"{path_folder}/test/wavs_sliced", f"{path_folder}/test/jukebox_feats")
            
        # slice motions/music into sliding windows to create training dataset
        print("Slicing train data")
        
        inpainting_frame = opt.inpainting_frame
        motion_len = opt.motion_len
        slice_len = motion_len*2
        
        slice_aistpp(f"{path_folder}/train/motions", f"{path_folder}/train/music_npy", 0.5, slice_len, inpainting_frame, motion_len)
        print("Slicing test data")
        slice_aistpp(f"{path_folder}/test/motions", f"{path_folder}/test/music_npy", 0.5, slice_len, inpainting_frame, motion_len)
    else:
        print("Slicing train data")
        slice_aistpp(f"{path_folder}/train/motions", f"{path_folder}/train/music_npy", 0.5, slice_len, inpainting_frame, motion_len)
        print("Slicing test data")
        slice_aistpp(f"{path_folder}/test/motions", f"{path_folder}/test/music_npy", 0.5, slice_len, inpainting_frame, motion_len)
    

def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="aistpp",
        choices=['aistpp', 'finedance'],
        help="name of dataset",
    )
    parser.add_argument(
        "--datapath",
        type=str,
        default="/raid/nhdang/Vy/data",
        help="path to folder containing motions and music",
    )
    parser.add_argument(
        "--inpainting_frame",
        type=float,
        default=1.0,
        help="the length (in seconds) of inpainting transition between 2 motions",
    )
    parser.add_argument(
        "--motion_len",
        type=float,
        default=3.0,
        help="each motion segment's length",
    )
    parser.add_argument("--extract-baseline", action="store_true")
    parser.add_argument("--extract-jukebox", action="store_true")
    opt = parser.parse_args()
    return opt


if __name__ == "__main__":
    opt = parse_opt()
    create_dataset(opt)