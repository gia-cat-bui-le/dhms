from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader, get_motion_loader
from data_loaders.humanml.utils.metrics import *
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel

from vis import SMPLSkeleton
from data_loaders.d2m.quaternion import ax_from_6v

from scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
from scipy import linalg

import pickle
from pathlib import Path

from evaluation.features.kinetic import extract_kinetic_features
from evaluation.features.manual_new import extract_manual_features

from data_loaders.d2m.audio_extraction.jukebox_features import extract as jukebox_extract
from data_loaders.d2m.audio_extraction.baseline_features import extract as baseline_extract

import scipy 
from scipy.io import wavfile 

import glob
import multiprocessing
import blobfile as bf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Any

from data_loaders.tensors import collate_generate

torch.multiprocessing.set_sharing_strategy('file_system')

class GenerateDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        feature_type: str = "baseline",
        normalizer: Any = None,
    ):
        self.data_path = data_path
        # print(self.data_path)
        self.raw_fps = 60
        self.data_fps = 30
        assert self.data_fps <= self.raw_fps
        self.data_stride = self.raw_fps // self.data_fps

        self.feature_type = feature_type

        self.normalizer = normalizer

        backup_path = os.path.join(data_path, "dataset_backups")
        Path(backup_path).mkdir(parents=True, exist_ok=True)
        
        # load raw data
        print("Loading dataset...")
        data = self.load_data()  # Call this last

        self.data = data
        self.length = len(data["filenames"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        feature = torch.from_numpy(np.load(filename_))
        num_feats, feature_slice = slice_audio(filename_, 3.0)
        feature_slice = torch.from_numpy(feature_slice)
        
        data_length = 90
        length_transistion = 60
        
        print("FEATURE SHAPE: ", feature_slice.feature_slice,
            "LENGTH SHAPE: ", data_length.shape)
        return {
            "length": data_length,
            "length_transition": length_transistion,
            "music": feature_slice,
            "filename": filename_
        }

    def load_data(self):
        # open data path
        sound_path = os.path.join(self.data_path, f"feature")
        # sort motions and sounds
        features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        data = {"filenames": features}
        return data

def slice_audio(audio_file, length):
    # stride, length in seconds
    FPS = 30
    audio = np.load(audio_file)
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * FPS)
    audio_slices = []
    while start_idx <= len(audio):
        if start_idx + window <= len(audio):
            audio_slice = audio[start_idx : start_idx + window]
        else:
            missing_length = window - (len(audio) - start_idx)
            missing_audio_slice = np.zeros(missing_length)
            audio_slice = np.concatenate((audio[start_idx:], missing_audio_slice))
            
        audio_slices.append(audio_slice)
        start_idx += window
        idx += 1
        
    return idx, audio_slices

def get_audio_length(audio_file_path):
    try:
        sample_rate, data = wavfile.read(audio_file_path)
        len_data = len(data)
        length_sec = int(len_data / sample_rate)
        return length_sec
    except Exception as e:
        print(f"Error: {e}")
        return None

def parse_resume_step_from_filename(filename):
    """
    Parse filenames of the form path/to/modelNNNNNN.pt, where NNNNNN is the
    checkpoint's number of steps.
    """
    split = filename.split("model")
    if len(split) < 2:
        return 0
    split1 = split[-1].split(".")[0]
    try:
        return int(split1)
    except ValueError:
        return 0

def get_dataset(args):
    DATA = GenerateDataset
    
    step = parse_resume_step_from_filename(args.model_path)
        
    normalizer_checkpoint = bf.join(
        bf.dirname(args.model_path), f"normalizer-{step:09}.pt"
    )
        
    checkpoint = torch.load(normalizer_checkpoint)
    loaded_normalizer = checkpoint["normalizer"]
        
    dataset = DATA(
    data_path=args.data_dir,
    normalizer=loaded_normalizer
    )
    
    return dataset

def get_dataset_loader(args, batch_size):
    dataset = get_dataset()
    num_cpus = multiprocessing.cpu_count()
    
    collate = collate_generate
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate
    )
    
    return loader, dataset.normalizer

if __name__ == '__main__':
    
    args = evaluation_parser()
    fixseed(args.seed)
    #TODO: fix the hardcode
    music_dir_len = len(os.listdir(args.music_dir))
    if music_dir_len > 32:
        args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    else:
        args.batch_size = music_dir_len
    name = os.path.basename(os.path.dirname(args.music_dir))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.music_dir), 'log_out', 'inference_{}_{}'.format(name, niter))
    
    if args.guidance_param != 1.:
        log_file += f'_gscale{args.guidance_param}'
    log_file += f'_inpaint{args.inpainting_frames}'
    if args.refine:
        log_file += f'_refine{args.refine_scale}'
    log_file += f'_comp{args.inter_frames}'
    log_file += f'_{args.eval_mode}'
    log_file += '.log'
    print(f'Will save to log file [{log_file}]')
    
    ########################################################################
    # LOAD SMPL
    
    smpl = SMPLSkeleton(device="cpu")
    
    ########################################################################

    print(f'Eval mode [{args.eval_mode}]')
    if args.eval_mode == 'debug':
        num_samples_limit = None  # None means no limit (eval over all dataset)
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 1  # about 3 Hrs
    elif args.eval_mode == 'wo_mm':
        num_samples_limit = 1000
        run_mm = False
        mm_num_samples = 0
        mm_num_repeats = 0
        mm_num_times = 0
        diversity_times = 300
        replication_times = 10# about 12 Hrs
    elif args.eval_mode == 'mm_short':
        num_samples_limit = 1000
        run_mm = True
        mm_num_samples = 100
        mm_num_repeats = 30
        mm_num_times = 10
        diversity_times = 300
        replication_times = 5  # about 15 Hrs
    else:
        raise ValueError()

    dist_util.setup_dist(args.device)
    logger.configure()
    
    #################### DATA LOADERS ###########################
    
    # extract feature from each music file 
    for music_file in os.listdir(args.music_dir):
        music_path = os.path.join(args.music_dir, music_file)
        baseline_extract(music_path, dest_dir=os.path.join(args.music_dir, "feature"))

    logger.log("creating data loader...")
    split = False
    origin_loader, _ = get_dataset_loader(args, batch_size=args.batch_size)
    
    num_actions = 1

    logger.log("Creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, origin_loader)

    logger.log(f"Loading checkpoints from [{args.model_path}]...")
    state_dict = torch.load(args.model_path, map_location='cpu')
    load_model_wo_clip(model, state_dict)

    if args.guidance_param != 1:
        model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
    model.to("cuda:0" if torch.cuda.is_available() else "cpu")
    model.eval()  # disable random masking

    eval_motion_loaders = {
        ################
        ## HumanML3D Dataset##
        ################
        'vald': lambda: get_mdm_loader(
            args, model, diffusion, args.batch_size,
            origin_loader, mm_num_samples, mm_num_repeats, num_samples_limit, args.guidance_param
        )
    }

    inference(eval_motion_loaders, origin_loader, args.out_dir, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)