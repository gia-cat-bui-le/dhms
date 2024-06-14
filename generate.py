from utils.parser_util import evaluation_parser, generate_args
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import (
    get_mdm_loader,
    get_motion_loader,
)
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

from data_loaders.d2m.audio_extraction.jukebox_features import (
    extract_folder as jukebox_extract,
)
from data_loaders.d2m.audio_extraction.baseline_features import (
    extract_folder as baseline_extract,
)

import scipy
from scipy.io import wavfile

import glob
import multiprocessing
import blobfile as bf
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from typing import Any

from teach.data.tools import lengths_to_mask
import random
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from data_loaders.d2m.quaternion import ax_from_6v, quat_slerp

torch.multiprocessing.set_sharing_strategy("file_system")
from pytorch3d.transforms import (
    RotateAxisAngle,
    axis_angle_to_quaternion,
    quaternion_multiply,
    quaternion_to_axis_angle,
)
from data_loaders.d2m.finedance.render_joints.smplfk import (
    SMPLX_Skeleton,
    do_smplxfk,
    ax_to_6v,
    ax_from_6v,
)
from diffusion.gaussian_diffusion import GaussianDiffusion
from copy import deepcopy
from utils.model_util import create_gaussian_diffusion
from data_loaders.d2m.preprocess import Normalizer, vectorize_many
from typing import List, Dict

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas

def collate_pairs_and_text(lst_elements: List, ) -> Dict:
    batch = {"motion_feats": collate_tensors([el["pose"] for el in lst_elements]),
            "filename": [x["filename"] for x in lst_elements],
            }
    return batch

class OriginDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        num_feats: list,
        feature_type: str = "baseline",
        normalizer: Any = None,
    ):
        self.data_path = data_path
        self.num_feats = num_feats
        self.feature_type = feature_type
        self.normalizer = normalizer

        # load raw data
        print("Loading dataset...")
        data = self.load_data()  # Call this last

        self.data = {
            "pos": data["pos"], 
            "q": data["q"],
            "filenames": data["filenames"],
        }
        
        self.length = len(data["pos"])

    def __len__(self):
        return self.length
    
    def process_dataset(self, root_pos, local_q):
        # FK skeleton
        smpl = SMPLSkeleton()
        # to Tensor
        root_pos = torch.Tensor(root_pos).unsqueeze(0)
        local_q = torch.Tensor(local_q).unsqueeze(0)
        
        root_pos = root_pos[:, :: 2, :]
        local_q = local_q[:, :: 2, :]
        
        # to ax
        bs, sq, c = local_q.shape
        # print(local_q.shape)
        local_q = local_q.reshape((bs, sq, -1, 3))

        # AISTPP dataset comes y-up - rotate to z-up to standardize against the pretrain dataset
        root_q = local_q[:, :, :1, :]  # sequence x 1 x 3
        root_q_quat = axis_angle_to_quaternion(root_q)
        rotation = torch.Tensor(
            [0.7071068, 0.7071068, 0, 0]
        )  # 90 degrees about the x axis
        root_q_quat = quaternion_multiply(rotation, root_q_quat)
        root_q = quaternion_to_axis_angle(root_q_quat)
        local_q[:, :, :1, :] = root_q

        # don't forget to rotate the root position too 😩
        pos_rotation = RotateAxisAngle(90, axis="X", degrees=True)
        root_pos = pos_rotation.transform_points(
            root_pos
        )  # basically (y, z) -> (-z, y), expressed as a rotation for readability

        # do FK
        positions = smpl.forward(local_q, root_pos)  # batch x sequence x 24 x 3
        feet = positions[:, :, (7, 8, 10, 11)]
        feetv = torch.zeros(feet.shape[:3])
        feetv[:, :-1] = (feet[:, 1:] - feet[:, :-1]).norm(dim=-1)
        contacts = (feetv < 0.01).to(local_q)  # cast to right dtype

        # to 6d
        local_q = ax_to_6v(local_q)

        # now, flatten everything into: batch x sequence x [...]
        l = [contacts, root_pos, local_q]
        global_pose_vec_input = vectorize_many(l).float().detach()

        assert not torch.isnan(global_pose_vec_input).any()

        print(f"Dataset Motion Features Dim: {global_pose_vec_input.shape}")

        return global_pose_vec_input

    def __getitem__(self, idx):
        
        print("numfeats: ", self.num_feats[idx])
        
        pose_input = self.process_dataset(self.data["pos"][idx][:2*self.num_feats[idx]], self.data["q"][idx][:2*self.num_feats[idx]])
        
        filename_ = self.data["filenames"][idx]
        
        return {
            "pose": pose_input,
            "filename": filename_
        }

    def load_data(self):
        # open data path
        motion_path = os.path.join(self.data_path)
        # sort motions and sounds
        motions = sorted(glob.glob(os.path.join(motion_path, "*.pkl")))
        
        all_pos = []
        all_q = []
        all_names = []
        
        for motion in motions:
            data = pickle.load(open(motion, "rb"))
            print(torch.Tensor(data["q"]).shape)
            pos = data["pos"]
            q = data["q"]
            scale = data["scale"][0]
            pos /= scale
            
            all_pos.append(pos)
            all_q.append(q)
            all_names.append(motion)

        data = {"pos": all_pos, "q": all_q, "filenames": all_names}
        return data

class GenerateDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        file_name,
        feature_type: str = "baseline",
        normalizer: Any = None,
    ):
        self.data_path = data_path
        # print(self.data_path)

        self.feature_type = feature_type
        self.file_name = file_name

        self.normalizer = normalizer

        # load raw data
        print("Loading dataset...")
        data = self.load_data()  # Call this last

        self.data = data
        self.length = len(data["filenames"])

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        filename_ = self.data["filenames"][idx]
        
        num_feats, feature_slice = slice_audio(filename_, 3.0)
        
        feature_slice = torch.from_numpy(feature_slice)

        data_length = 90

        print("FEATURE SHAPE: ", feature_slice.shape)
        bs, seq, d = feature_slice.shape
        return {"length": data_length, "music": feature_slice.reshape(bs, seq * d), "filename": filename_}

    def load_data(self):
        # open data path
        sound_path = os.path.join(self.data_path, "feature")
        # sort motions and sounds
        # features = sorted(glob.glob(os.path.join(sound_path, "*.npy")))
        features = [str(self.file_name)]
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
    while start_idx + window <= len(audio):
        audio_slice = audio[start_idx : start_idx + window]
        # if start_idx + window <= len(audio):
        #     audio_slice = audio[start_idx : start_idx + window]
        # else:
        #     missing_length = window - (len(audio) - start_idx)
        #     missing_audio_slice = np.zeros((missing_length, 4800), dtype=np.float32)
        #     audio_slice = np.concatenate((audio[start_idx:], missing_audio_slice))

        audio_slices.append(audio_slice)
        start_idx += window
        idx += 1

    return idx, np.array(audio_slices)


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


def get_dataset(args, file_name):
    DATA = GenerateDataset

    step = parse_resume_step_from_filename(args.model_path)
    dataset = DATA(data_path=os.path.join(args.music_dir), file_name=file_name, normalizer=None)

    return dataset


def get_dataset_loader(args, file_name, batch_size):
    dataset = get_dataset(args, file_name)
    num_cpus = multiprocessing.cpu_count()

    print(f"batchsize: {batch_size}")

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=min(int(num_cpus * 0.75), 32),
        pin_memory=True,
        drop_last=True,
        # collate_fn=collate
    )

    return loader, dataset.normalizer

if __name__ == "__main__":
    args = generate_args()
    fixseed(args.seed)
    # TODO: fix the hardcode
    
    music_dir_len = len(glob.glob(os.path.join(args.music_dir, "*.wav")))
    print(f"music dir len {music_dir_len}")
    # if music_dir_len > 32:
    #     args.batch_size = 32  # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    # else:
    #     args.batch_size = 4
    args.batch_size = music_dir_len
    name = os.path.basename(os.path.dirname(args.music_dir))
    niter = os.path.basename(args.model_path).replace("model", "").replace(".pt", "")
    log_file = os.path.join(
        os.path.dirname(args.music_dir),
        "log_out",
        "inference_{}_{}".format(name, niter),
    )

    if args.guidance_param != 1.0:
        log_file += f"_gscale{args.guidance_param}"
    log_file += f"_inpaint{args.inpainting_frames}"
    if args.refine:
        log_file += f"_refine{args.refine_scale}"
    log_file += f"_comp{args.inter_frames}"
    log_file += ".log"
    print(f"Will save to log file [{log_file}]")

    ########################################################################
    # LOAD SMPL
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    smpl = SMPLSkeleton(device=device)

    ########################################################################

    num_samples_limit = None  # None means no limit (eval over all dataset)
    run_mm = False
    mm_num_samples = 0
    mm_num_repeats = 0
    mm_num_times = 0
    diversity_times = 300
    replication_times = 1  # about 3 Hrs

    dist_util.setup_dist(args.device)
    logger.configure()

    #################### DATA LOADERS ###########################

    # extract feature from each music file
    #! code for jukebox
    # baseline_extract(args.music_dir, dest=os.path.join(args.music_dir, "feature"))

    logger.log("creating data loader...")
    split = False

    #!: mỗi batch = 1 bài hát. vd có 2 bài => origin loader gồm 2 batch, mỗi batch sẽ chứa feature được sliced ra từ bài hát load vào
    
    if args.dataset == "aistpp":
        nfeats = 151
        njoints = 24
        smpl = SMPLSkeleton(device=device)
    elif args.dataset == "finedance":
        nfeats = 139
        njoints = 22
        smpl = SMPLX_Skeleton(device=device, Jpath="data_loaders/d2m/body_models/smpl/smplx_neu_J_1.npy")

    scale = 1
    
    file_names = sorted(list(Path(os.path.join(args.music_dir, "feature")).glob("*.npy")))
    
    
    generate_len = []
    
    for file_name in file_names:
        
        dataloader, _ = get_dataset_loader(args, file_name, batch_size=1)
        
        num_actions = 1

        logger.log("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, dataloader)

        logger.log(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location="cpu")
        load_model_wo_clip(model, state_dict)

        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(
                model
            )  # wrapping model with the classifier-free sampler

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model.to(device)
        model.eval()  # disable random masking

        generated_motion = []
        mm_generated_motions = []
        clip_denoised = False  #! hardcoded (from repo)
        if args.refine:
            sample_fn_refine = diffusion.p_sample_loop

        print(f"diffusion: {diffusion}")
        composition = args.composition
        use_ddim = False  # hardcode

        sample_fn = diffusion.p_sample_loop

        print(f"composition: {composition}, sample fn: {sample_fn}")

        with torch.no_grad():
            for _, batch in tqdm(enumerate(dataloader)):
                print("filename len: ", len(batch["filename"]))
                for i in range(len(batch["filename"])):
                    batch_music = batch["music"][i]
                    batch_filename = batch["filename"][i]
                    batch_length = batch["length"][i]
                    
                    print(f"batch shape: {batch_music.shape}")
                    if (
                        num_samples_limit is not None
                        and len(generated_motion) >= num_samples_limit
                    ):
                        print("if num samples limit")
                        break
                        
                    bs, music_dim = batch_music.shape
                    device = "cuda:0" if torch.cuda.is_available() else "cpu"
                    
                    model_kwargs = {}
                    model_kwargs['y'] = {}
                    model_kwargs['y']['lengths'] = [90 for len in range(bs)]
                    model_kwargs['y']['music'] = batch_music.to("cuda:0" if torch.cuda.is_available() else "cpu")
                    model_kwargs['y']['mask'] = lengths_to_mask(model_kwargs['y']['lengths'], 
                                        dist_util.dev()).unsqueeze(1).unsqueeze(2)
                    
                    if scale != 1.:
                        model_kwargs['y']['scale'] = torch.ones(len(model_kwargs['y']['lengths']),
                                                                device="cuda:0" if torch.cuda.is_available() else "cpu") * scale
                        
                    mm_num_now = len(mm_generated_motions) // dataloader.batch_size
                    is_mm = False
                    
                    sample = diffusion.p_sample_loop (
                        model,
                        (bs, nfeats, 1, model_kwargs['y']['mask'].shape[-1]),
                        noise=None,
                        clip_denoised=clip_denoised,
                        model_kwargs=model_kwargs,
                        skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                        init_image=None,
                        progress=False,
                        dump_steps=None,
                        const_noise=False,
                        # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                    )
                    
                    motion_final = []
                    motion_final.append(sample[0].squeeze().unsqueeze(dim=0).permute(0, 2, 1))
                    
                    for idx in range(bs - 1):
                        sample_0 = sample[idx].unsqueeze(0)
                        sample_1 = sample[idx + 1].unsqueeze(0)
                        
                        print(f"shape check:\n\tsample 0: {sample_0.shape}\n\tsample 1: {sample_1.shape}")
                        
                        music_0 = model_kwargs['y']['music'][idx, -45 * 4800:].unsqueeze(0)
                        music_1 = model_kwargs['y']['music'][idx + 1, :45 * 4800].unsqueeze(0)
                        
                        print(f"shape check:\n\tmusic 0: {music_0.shape}\n\tmusic 1: {music_1.shape}")
                    
                        num_rows = 1
                        motion = torch.cat(( sample_0[:, :, :, -45 :], sample_1[:, :, :, : 45]), -1)
                        assert motion.shape == (1, nfeats, 1, 90)
                        input_motions = motion.repeat((num_rows, 1, 1, 1))
                        
                        max_frames = input_motions.shape[-1]
                        assert max_frames == input_motions.shape[-1]
                        gt_frames_per_sample = {}
                        
                        model_kwargs_2 = {}
                        model_kwargs_2['y'] = {}

                        model_kwargs_2['y']['lengths'] = [90 for len in range(1)]
                        model_kwargs_2['y']['music'] = torch.cat((music_0, music_1), dim=1).to("cuda:0" if torch.cuda.is_available() else "cpu")
                        model_kwargs_2['y']['mask'] = lengths_to_mask(model_kwargs_2['y']['lengths'], 
                                            dist_util.dev()).unsqueeze(1).unsqueeze(2)
                        # add CFG scale to batch
                        if scale != 1.:
                            model_kwargs_2['y']['scale'] = torch.ones(len(model_kwargs_2['y']['lengths']),
                                                                    device="cuda:0" if torch.cuda.is_available() else "cpu") * scale
                        
                        total_hist_frame = args.inpainting_frames + 15
                        hist_lst = [feats[:,:,:90] for feats in sample_0]
                        hframes = torch.stack([x[:,:,-total_hist_frame : -15] for x in hist_lst])
                        
                        fut_lst = [feats[:,:,:90] for feats in sample_1]
                        fut_frames = torch.stack([x[:,:,15:total_hist_frame] for x in fut_lst])

                        model_kwargs_2['y']['hframes'] = hframes
                        model_kwargs_2['y']['fut_frames'] = fut_frames
                            
                        model_kwargs_2['y']['inpainting_mask'] = torch.ones_like(input_motions, dtype=torch.float,
                                                                    device=input_motions.device)  # True means use gt motion
                        for i, length in enumerate(model_kwargs_2['y']['lengths']):
                            start_idx, end_idx = 30, 60
                            gt_frames_per_sample[i] = list(range(0, start_idx)) + list(range(end_idx, max_frames))
                            model_kwargs_2['y']['inpainting_mask'][i, :, :, start_idx: end_idx] = False  # do inpainting in those frames
                            mask_slope = 10
                            for f in range(mask_slope):
                                if start_idx-f < 0:
                                    continue
                                model_kwargs_2['y']['inpainting_mask'][i, :, :, start_idx-f] = f/mask_slope
                                if end_idx+f >= length:
                                    continue
                                model_kwargs_2['y']['inpainting_mask'][i, :, :, end_idx+f] = f/mask_slope
                        
                        sample_2 = diffusion.p_sample_loop (
                            model,
                            (1, nfeats, 1, model_kwargs_2['y']['mask'].shape[-1]),
                            noise=None,
                            clip_denoised=clip_denoised,
                            model_kwargs=model_kwargs_2,
                            skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                            init_image=None,
                            progress=False,
                            dump_steps=None,
                            const_noise=False,
                            # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                        )
                        
                        assert sample_0.shape == sample_1.shape == sample_2.shape == (1, nfeats, 1, 90)
                        
                        motion_final.append(sample_2.squeeze().unsqueeze(dim=0).permute(0, 2, 1))
                        motion_final.append(sample_1.squeeze().unsqueeze(dim=0).permute(0, 2, 1))
                        
                    motion_result = torch.cat(motion_final, dim=0)
                    if motion_result.shape[2] == nfeats:
                        sample_contact, motion_result = torch.split(
                            motion_result, (4, motion_result.shape[2] - 4), dim=2
                        )
                    else:
                        sample_contact = None
                        # do the FK all at once
                    
                    b, s, c_ = motion_result.shape
                    pos = motion_result[:, :, :3].to(device)  # np.zeros((sample.shape[0], 3))
                    q = motion_result[:, :, 3:].reshape(b, s, njoints, 6)
                    # go 6d to ax
                    q = ax_from_6v(q).to(device)

                    b, s, c1, c2 = q.shape
                    assert s % 2 == 0
                    half = s // 2
                    assert half == 45
                    if b > 1:
                        # if long mode, stitch position using linear interp

                        fade_out = torch.ones((1, s, 1)).to(pos.device)
                        fade_in = torch.ones((1, s, 1)).to(pos.device)
                        fade_out[:, half:, :] = torch.linspace(1, 0, half)[None, :, None].to(
                            pos.device
                        )
                        fade_in[:, :half, :] = torch.linspace(0, 1, half)[None, :, None].to(
                            pos.device
                        )

                        pos[:-1] *= fade_out
                        pos[1:] *= fade_in

                        full_pos = torch.zeros((s + half * (b - 1), 3)).to(pos.device)
                        id_ = 0
                        for pos_slice in pos:
                            full_pos[id_ : id_ + s] += pos_slice
                            id_ += half

                        # stitch joint angles with slerp
                        slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(pos.device)

                        left, right = q[:-1, half:], q[1:, :half]
                        # convert to quat
                        left, right = (
                            axis_angle_to_quaternion(left),
                            axis_angle_to_quaternion(right),
                        )
                        merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
                        # convert back
                        merged = quaternion_to_axis_angle(merged)

                        full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(pos.device)
                        full_q[:half] += q[0, :half]
                        id_ = half
                        for q_slice in merged:
                            full_q[id_ : id_ + half] += q_slice
                            id_ += half
                        full_q[id_ : id_ + half] += q[-1, half:]
                        
                        full_pos = full_pos.unsqueeze(0)
                        full_q = full_q.unsqueeze(0)
                        
                        # assert full_pos.shape == (1, 180, 3)
                        # assert full_q.shape == (1, 180, njoints, 3)
                        
                        full_pose = (
                            smpl.forward(full_q, full_pos).squeeze(0).detach().cpu().numpy()
                        )  # b, s, 24, 3
                        
                        if njoints == 24:
                            # assert full_pose.shape == (180, njoints, 3)
                            assert full_pose.shape[1] == njoints
                        else:
                            # assert full_pose.shape == (180, 55, 3)
                            assert full_pose.shape[1] == 55
                        
                        filename = batch_filename
                        outname = f'{args.inference_dir}/inference/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
                        out_path = os.path.join("./", outname)
                        print(out_path)
                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        # print(out_path)
                        generate_len.append(full_pose.squeeze().shape[0])
                        print("Generate shape before trim: ", full_pose.squeeze().shape)
                        # full_pose = full_pose[:210]
                        # print("Generate shape after trim: ", full_pose.squeeze().shape)
                        
                        with open(out_path, "wb") as file_pickle:
                            pickle.dump(
                                {
                                    "smpl_poses": full_q.squeeze(0).reshape((-1, njoints * 3)).cpu().numpy(),
                                    "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                                    "full_pose": full_pose.squeeze(),
                                },
                                file_pickle,
                            )
                            
    # origin_dataset = OriginDataset(
    #     data_path=os.path.join(args.music_dir, "motions"), num_feats=generate_len, normalizer=None
    # )
    # origin_loader = DataLoader(
    #     origin_dataset,
    #     batch_size=1,
    #     shuffle=False,
    #     num_workers=min(int(multiprocessing.cpu_count() * 0.75), 32),
    #     pin_memory=True,
    #     drop_last=True,
    #     collate_fn=collate_pairs_and_text
    # )
    
    # print(len(origin_loader))
    
    # for batch in origin_loader:
    #     njoints = 24
    #     smpl = SMPLSkeleton(device=device)
        
    #     motion, filenames = batch["motion_feats"][0], batch["filename"][0]
    #     motion = torch.Tensor(motion).to(device)
        
    #     print(motion.shape)
    #     b, s, c = motion.shape
        
    #     sample_contact, motion = torch.split(
    #     motion, (4, motion.shape[2] - 4), dim=2)
    #     pos = motion[:, :, :3].to(motion.device)  # np.zeros((sample.shape[0], 3))
    #     q = motion[:, :, 3:].reshape(b, s, njoints, 6)
    #     # go 6d to ax
    #     q = ax_from_6v(q).to(motion.device)
        
    #     for q_, pos_ in zip(q, pos):
            
    #         # if out_dir is not None:
    #         print("GT shape: ", (smpl.forward(q_.unsqueeze(0), pos_.unsqueeze(0))).squeeze(0).shape)
    #         full_pose = (smpl.forward(q_.unsqueeze(0), pos_.unsqueeze(0)).squeeze(0).detach().cpu().numpy())
    #         outname = f'{args.inference_dir}/gt/{"".join(os.path.splitext(os.path.basename(filenames)))}.pkl'
    #         out_path = os.path.join(outname)
    #         # Create the directory if it doesn't exist
    #         os.makedirs(os.path.dirname(out_path), exist_ok=True)
    #         with open(out_path, "wb") as file_pickle:
    #             pickle.dump(
    #                 {
    #                     "smpl_poses": q_.squeeze(0).reshape((-1, njoints * 3)).cpu().numpy(),
    #                     "smpl_trans": pos_.squeeze(0).cpu().numpy(),
    #                     "full_pose": full_pose,
    #                 },
    #                 file_pickle,
    #             )