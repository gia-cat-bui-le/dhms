# This code is based on https://github.com/openai/guided-diffusion
"""
Generate a large batch of image samples from a model and save them as a large
numpy array. This can be used to produce samples for FID evaluation.
"""
from utils.fixseed import fixseed
import os
import numpy as np
import torch
from utils.parser_util import generate_args
from utils.model_util import create_model_and_diffusion, load_model_wo_clip
from utils import dist_util
from model.cfg_sampler import ClassifierFreeSampleModel
from data_loaders.get_data import get_dataset_loader
from data_loaders.humanml.scripts.motion_process import recover_from_ric
import data_loaders.humanml.utils.paramUtil as paramUtil
from data_loaders.humanml.utils.plot_script import plot_3d_motion
import shutil
from data_loaders.tensors import collate
from torch.utils.data import DataLoader
from vis import SMPLSkeleton, skeleton_render
import pickle

import multiprocessing
# data_loaders\d2m\aistpp\audio_extraction\baseline_features.py
from data_loaders.d2m.audio_extraction.baseline_features import extract as baseline_extract
from data_loaders.d2m.audio_extraction.jukebox_features import extract as juke_extract

import glob
import os
from functools import cmp_to_key
from pathlib import Path
from tempfile import TemporaryDirectory
import random
from torch.utils.data import Dataset
from diffusion import logger

import jukemirlib
from tqdm import tqdm
from teach.data.tools import lengths_to_mask 

from data_loaders.d2m.slice import slice_audio
from scaler import MinMaxScaler

from accelerate import Accelerator, DistributedDataParallelKwargs

from pytorch3d.transforms import (axis_angle_to_quaternion,
                                  quaternion_to_axis_angle)

from data_loaders.d2m.quaternion import ax_from_6v, quat_slerp

#TODO: code lại Dataset Class --> mỗi data nhận full nhạc & chia thành nhiều segment; Dataset chứa info tổng len

key_func = lambda x: int(os.path.splitext(x)[0].split("_")[-1].split("slice")[-1])

def stringintcmp_(a, b):
    aa, bb = "".join(a.split("_")[:-1]), "".join(b.split("_")[:-1])
    ka, kb = key_func(a), key_func(b)
    if aa < bb:
        return -1
    if aa > bb:
        return 1
    if ka < kb:
        return -1
    if ka > kb:
        return 1
    return 0


stringintkey = cmp_to_key(stringintcmp_)

class GenerateDataset(Dataset):
    def __init__(
        self,
        cond, 
        n_frames
    ):
        self.length = len(cond)
        self.cond = cond
        self.n_frames = n_frames

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        feature = self.cond[idx]
        seq, d = feature.shape
        print("feature shape: ", feature.shape, "n frame: ", self.n_frames)
        return {
            'music': feature,
            'n_frames': self.n_frames
        }


# def lengths_to_mask(lengths: List[int], device: torch.device) -> Tensor:
#     lengths = torch.tensor(lengths, device=device)
#     max_len = max(lengths)
#     mask = torch.arange(max_len, device=device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
#     return mask

def generate_collate(batch):
    # print(batch[0]["pose"])
    # exit()
    notnone_batches = [b for b in batch if b is not None]
    lenbatch = [len(b['music']) for b in notnone_batches]
    # print(lenbatch)
    lenmotion = [b['n_frames'] for b in notnone_batches]
    
    lenmotionTensor = torch.as_tensor(lenmotion)
    lenbatchTensor = torch.as_tensor(lenbatch)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, lenbatch[0]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting
    
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenmotionTensor}}
    
    if "music" in notnone_batches[0]:
        musicbatch = [b['music'].tolist()[:] for b in notnone_batches]
        cond['y'].update({
            'music': torch.tensor(np.array(musicbatch), dtype=torch.float32)
            })

    return cond

class Normalizer:
    def __init__(self, data):
        flat = data.reshape(-1, data.shape[-1])
        self.scaler = MinMaxScaler((-1, 1), clip=True)
        self.scaler.fit(flat)

    def normalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        return self.scaler.transform(x).reshape((batch, seq, ch))

    def unnormalize(self, x):
        batch, seq, ch = x.shape
        x = x.reshape(-1, ch)
        x = torch.clip(x, -1, 1)  # clip to force compatibility
        return self.scaler.inverse_transform(x).reshape((batch, seq, ch))
        
def main():
    args = generate_args()
    args.batch_size = 32
    fixseed(args.seed)
    out_path = args.output_dir
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    max_frames = 1800
    fps = 30
    n_frames = min(max_frames, int(args.motion_length*fps))
    
    if out_path == '':
        out_path = os.path.join(os.path.dirname(args.model_path),
                                'samples_{}_{}_seed{}'.format(name, niter, args.seed))
        
    feature_func = juke_extract if args.feature_type == "jukebox" else baseline_extract
    
    #TODO: sample_length will be extracted from music's length
    sample_length = args.motion_length
    # sample_size = int(sample_length / 5.0) - 1
    # print(sample_size)
    
    temp_dir_list = []
    all_cond = []
    all_filenames = []
    
    if args.custom_input:
        print("Computing features for input music")
        for wav_file in glob.glob(os.path.join(args.music_dir, "*.wav")):
            # create temp folder (or use the cache folder if specified)
            if args.cache_features:
                songname = os.path.splitext(os.path.basename(wav_file))[0]
                save_dir = os.path.join(args.feature_cache_dir, songname)
                Path(save_dir).mkdir(parents=True, exist_ok=True)
                dirname = save_dir
            else:
                temp_dir = TemporaryDirectory()
                temp_dir_list.append(temp_dir)
                dirname = temp_dir.name
                
            mum_slice = slice_audio(wav_file, 5.0, 5.0, dirname)
            if mum_slice == 0:
                try:
                    os.rmdir(dirname)
                    print(f"Folder '{dirname}' removed successfully.")
                except OSError as e:
                    print(f"Error: {e.strerror}")
            else:
                file_list = sorted(glob.glob(f"{dirname}/*.wav"), key=stringintkey)
                cond_list = []
                for idx, file in enumerate(tqdm(file_list)):
                    reps, _ = feature_func(file)
                    # save reps
                    if args.cache_features:
                        featurename = os.path.splitext(file)[0] + ".npy"
                        np.save(featurename, reps)
                    cond_list.append(reps)
            cond_list = np.array(cond_list)
            all_cond.append(cond_list)
            all_filenames.append(file_list)
    else:
        print("Using test data")
        if args.dataset == "aistpp":
            test_path = os.path.join(args.test_dir, "aistpp_dataset")
        else:
            test_path = os.path.join(args.test_dir, "finedance")
            
        print(args.dataset, test_path)
        test_path = os.path.join(test_path, "test/music_npy_sliced")
        
        file_ids = set(file_name.split('_')[0] for file_name in os.listdir(test_path))
        
        for file_id in file_ids:
            cond_list = []
            file_list = []
            for music_file in os.listdir(test_path):
                if music_file.startswith(file_id):
                    file_list.append(os.path.join(test_path, music_file))
                    cond_list.append(np.load(os.path.join(test_path, music_file)))
            cond_list = np.array(cond_list)
            all_cond.append(cond_list)
            all_filenames.append(file_list)
    
    for index, each_cond in tqdm(enumerate(all_cond)):
        # print(each_cond.shape)
        dataset = GenerateDataset(cond=each_cond, n_frames=n_frames)
        
        num_cpus = multiprocessing.cpu_count()
        real_num_batches = len(dataset)
        print('real_num_batches', real_num_batches)
        
        dataloader = DataLoader(
            dataset,
            batch_size=real_num_batches,
            shuffle=False,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=False,
            collate_fn=generate_collate
        )
                
        mm_num_samples = args.num_samples
        mm_num_repeats = args.num_repetitions
                
        assert args.num_samples <= args.batch_size, \
            f'Please either increase batch_size({args.batch_size}) or reduce num_samples({args.num_samples})'
            
        use_ddim = False  # FIXME - hardcoded
        clip_denoised = False  # FIXME - hardcoded
        composition = args.composition
        inter_frames = args.inter_frames
            
        # print('Loading dataset...')
        # data = load_dataset(args, max_frames, n_frames)
        # total_num_samples = args.num_samples * args.num_repetitions
        logger.log("Creating model and diffusion...")
        model, diffusion = create_model_and_diffusion(args, dataloader)

        logger.log(f"Loading checkpoints from [{args.model_path}]...")
        state_dict = torch.load(args.model_path, map_location='cpu')
        load_model_wo_clip(model, state_dict)
        
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print("GUIDANCE PARAM: ", args.guidance_param)
        if args.guidance_param != 1:
            model = ClassifierFreeSampleModel(model)   # wrapping model with the classifier-free sampler
        model.to(device)
        model.eval()
        
        if composition:
            sample_fn = diffusion.p_sample_loop_comp
        else:
            sample_fn = (
                diffusion.p_sample_loop_inpainting if not use_ddim else diffusion.ddim_sample_loop
            )
        if args.refine:
            sample_fn_refine = diffusion.p_sample_loop
        
        generated_motion = []
        mm_generated_motions = []
        if mm_num_samples > 0:
            mm_idxs = np.random.choice(real_num_batches, mm_num_samples // dataloader.batch_size +1, replace=False)
            mm_idxs = np.sort(mm_idxs)
        else:
            mm_idxs = []
        print('mm_idxs', mm_idxs)

        # model.eval()
        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
        accelerator = Accelerator(kwargs_handlers=[ddp_kwargs])
        
        smpl = None
        
        if args.dataset == "aistpp":
            noise_dim = 151
            smpl = SMPLSkeleton(accelerator.device)
        elif args.dataset == "finedance":
            noise_dim = 319
        
        with torch.no_grad():
            for data_index, batch in tqdm(enumerate(dataloader)):
                print("DATA: ", data_index)
                scale = 1.
                # if num_samples_limit is not None and len(generated_motion) >= num_samples_limit:
                #     break

                bs = len(batch['y']['lengths'])

        #         model_kwargs_0 = {}
        #         model_kwargs_0['y'] = {}
        #         if args.inter_frames > 0:
        #             model_kwargs_0['y']['lengths'] = [len + args.inter_frames // 2 for len in batch['y']['lengths']].to(device)
        #         else:
        #             model_kwargs_0['y']['lengths'] = batch['y']['lengths'].to(device)
        #         model_kwargs_0['y']['music'] = batch['y']['music'].to(device)
        #         model_kwargs_0['y']['mask'] = lengths_to_mask(model_kwargs_0['y']['lengths'], 
        #                             device=model_kwargs_0['y']['lengths'].device).unsqueeze(1).unsqueeze(2).to(device)
                
        #         model_kwargs_0['y']['scale'] = torch.ones(len(model_kwargs_0['y']['lengths']),
        #                                                 device=device).to(device) * scale

        #         mm_num_now = len(mm_generated_motions) // dataloader.batch_size
        #         is_mm = data_index in mm_idxs
        #         repeat_times = 1
        #         mm_motions = []
        #         # print("repeat time: ", repeat_times)
        #         print("NOISE DIM: ", noise_dim)
                
        #         noise_0 = None
        #         # noise_0 = torch.randn(
        #         #     [bs, noise_dim, model_kwargs_0['y']['mask'].shape[-1]], device=model_kwargs_0['y']['mask'].device
        #         #     )
        #         # window_size = 10
        #         # window_stride = 5
        #         # for frame_index in range(window_size, model_kwargs_0['y']['mask'].shape[-1], window_stride):
        #         #     list_index = list(
        #         #         range(
        #         #             frame_index - window_size,
        #         #             frame_index + window_stride - window_size,
        #         #         )
        #         #     )
        #         #     random.shuffle(list_index)
        #         #     noise_0[
        #         #         :, :, frame_index : frame_index + window_stride
        #         #     ] = noise_0[:, :, list_index]
                
        #         for t in tqdm(range(repeat_times)):
        #             sample_0 = sample_fn(
        #                 model,
        #                 args.hist_frames,
        #                 args.inpainting_frames if not args.composition else args.inter_frames,
        #                 (bs, noise_dim, model_kwargs_0['y']['mask'].shape[-1]),
        #                 # (bs, 151, model_kwargs_1['y']['mask'].shape[-1]),
        #                 clip_denoised=clip_denoised,
        #                 model_kwargs_0=model_kwargs_0,
        #                 # model_kwargs_1=model_kwargs_1,
        #                 skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
        #                 init_image=None,
        #                 progress=False,
        #                 dump_steps=None,
        #                 noise=noise_0,
        #                 const_noise=False,
        #                 # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
        #             )
                    
        #             samples = sample_0.permute(0, 2, 1).cpu() # B L D
                    
        #             nfeats = samples.shape[2]

        #             if samples.shape[2] == 319 or samples.shape[2] == 151 or samples.shape[2] == 139:                 # debug if samples.shape[2] == 151:    
        #                 sample_contact, samples = torch.split(
        #                     samples, (4, samples.shape[2] - 4), dim=2
        #                 )
        #             else:
        #                 sample_contact = None
        #             # do the FK all at once
        #             b, s, c = samples.shape
        #             pos = samples[:, :, :3].to(model_kwargs_0['y']['music'].device)  # np.zeros((sample.shape[0], 3))
        #             q = samples[:, :, 3:].reshape(b, s, -1, 6)
        #             # go 6d to ax
        #             q = ax_from_6v(q).to(model_kwargs_0['y']['music'].device)

        #             b, s, c1, c2 = q.shape
        #             # assert s % 2 == 0
        #             # half = s // 2
        #             if b > 1:

        #                 full_pos = torch.zeros((s * b, 3)).to(pos.device)
        #                 idx = 0
        #                 for pos_slice in pos:
        #                     full_pos[idx : idx + s] += pos_slice
        #                     idx += s

        #                 # stitch joint angles with slerp
        #                 slerp_weight = torch.linspace(0, 1, s)[None, :, None].to(pos.device)

        #                 left, right = q[:-1, :], q[1:, :]
        #                 # convert to quat
        #                 left, right = (
        #                     axis_angle_to_quaternion(left),
        #                     axis_angle_to_quaternion(right),
        #                 )
        #                 merged = quat_slerp(left, right, slerp_weight)  # (b-1) x half x ...
        #                 # convert back
        #                 merged = quaternion_to_axis_angle(merged)

        #                 full_q = torch.zeros((s * b, c1, c2)).to(pos.device)
        #                 full_q[:s] += q[0, :s]
        #                 idx = s
        #                 for q_slice in merged:
        #                     full_q[idx : idx + s] += q_slice
        #                     idx += s
        #                 full_q[idx : idx + s] += q[-1, s:]

        #                 # unsqueeze for fk
        #                 full_pos = full_pos.unsqueeze(0)
        #                 full_q = full_q.unsqueeze(0)
        #             else:
        #                 full_pos = pos
        #                 full_q = q
                        
        #             if nfeats == 151:
        #                 full_pose = (
        #                     smpl.forward(full_q, full_pos).detach().cpu().numpy()
        #                 )  # b, s, 24, 3

        #                 fk_out = "save/results"
        #                 # skeleton_render(
        #                 #     full_pose[0],
        #                 #     epoch=f"{index}",
        #                 #     out="save/renders",
        #                 #     name=all_filenames[index],
        #                 #     sound=True,
        #                 #     stitch=True,
        #                 #     sound_folder="cached_features",
        #                 #     render=True
        #                 # )
        #                 if fk_out is not None:
        #                     outname = f'{index}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'
        #                     Path(fk_out).mkdir(parents=True, exist_ok=True)
        #                     pickle.dump(
        #                         {
        #                             "smpl_poses": full_q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
        #                             "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
        #                             "full_pose": full_pose[0],
        #                         },
        #                         open(os.path.join(fk_out, outname), "wb"),
        #                     )
        #             elif nfeats == 319:
        #                 fk_out = "save2/results"
        #                 reshape_size = 156
        #                 outname = f'{index}_{"_".join(os.path.splitext(os.path.basename(name[0]))[0].split("_")[:-1])}.pkl'  # f'{epoch}_{"_".join(name)}.pkl' #    
        #                 Path(fk_out).mkdir(parents=True, exist_ok=True)
        #                 pickle.dump(
        #                     {
        #                         "smpl_poses": full_q.squeeze(0).reshape((-1, reshape_size)).cpu().numpy(),    # local rotations     
        #                         "smpl_trans": full_pos.squeeze(0).cpu().numpy(),                    # root translation
        #                         # "full_pose": full_pose[0],                                          # 3d positions
        #                     },
        #                     open(os.path.join(fk_out, outname), "wb"),
        #                 )


def save_multiple_samples(args, out_path, row_print_template, all_print_template, row_file_template, all_file_template,
                          caption, num_samples_in_out_file, rep_files, sample_files, sample_i):
    all_rep_save_file = row_file_template.format(sample_i)
    all_rep_save_path = os.path.join(out_path, all_rep_save_file)
    ffmpeg_rep_files = [f' -i {f} ' for f in rep_files]
    hstack_args = f' -filter_complex hstack=inputs={args.num_repetitions}' if args.num_repetitions > 1 else ''
    ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(ffmpeg_rep_files) + f'{hstack_args} {all_rep_save_path}'
    os.system(ffmpeg_rep_cmd)
    print(row_print_template.format(caption, sample_i, all_rep_save_file))
    sample_files.append(all_rep_save_path)
    if (sample_i + 1) % num_samples_in_out_file == 0 or sample_i + 1 == args.num_samples:
        # all_sample_save_file =  f'samples_{(sample_i - len(sample_files) + 1):02d}_to_{sample_i:02d}.mp4'
        all_sample_save_file = all_file_template.format(sample_i - len(sample_files) + 1, sample_i)
        all_sample_save_path = os.path.join(out_path, all_sample_save_file)
        print(all_print_template.format(sample_i - len(sample_files) + 1, sample_i, all_sample_save_file))
        ffmpeg_rep_files = [f' -i {f} ' for f in sample_files]
        vstack_args = f' -filter_complex vstack=inputs={len(sample_files)}' if len(sample_files) > 1 else ''
        ffmpeg_rep_cmd = f'ffmpeg -y -loglevel warning ' + ''.join(
            ffmpeg_rep_files) + f'{vstack_args} {all_sample_save_path}'
        os.system(ffmpeg_rep_cmd)
        sample_files = []
    return sample_files


def construct_template_variables(unconstrained):
    row_file_template = 'sample{:02d}.mp4'
    all_file_template = 'samples_{:02d}_to_{:02d}.mp4'
    if unconstrained:
        sample_file_template = 'row{:02d}_col{:02d}.mp4'
        sample_print_template = '[{} row #{:02d} column #{:02d} | -> {}]'
        row_file_template = row_file_template.replace('sample', 'row')
        row_print_template = '[{} row #{:02d} | all columns | -> {}]'
        all_file_template = all_file_template.replace('samples', 'rows')
        all_print_template = '[rows {:02d} to {:02d} | -> {}]'
    else:
        sample_file_template = 'sample{:02d}_rep{:02d}.mp4'
        sample_print_template = '["{}" ({:02d}) | Rep #{:02d} | -> {}]'
        row_print_template = '[ "{}" ({:02d}) | all repetitions | -> {}]'
        all_print_template = '[samples {:02d} to {:02d} | all repetitions | -> {}]'

    return sample_print_template, row_print_template, all_print_template, \
           sample_file_template, row_file_template, all_file_template


# def load_dataset(args, max_frames, n_frames):
#     data = get_dataset_loader(name=args.dataset,
#                               batch_size=args.batch_size,
#                               num_frames=max_frames,
#                               split='test',
#                               hml_mode='text_only')
#     if args.dataset in ['kit', 'humanml']:
#         data.dataset.t2m_dataset.fixed_length = n_frames
#     return data


if __name__ == "__main__":
    torch.multiprocessing.set_start_method('spawn')
    main()