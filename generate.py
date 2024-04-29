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

# NOTE: add imports
from data_loaders.humanml.motion_loaders.comp_v6_model_dataset import (
    create_pre_sample,
    create_post_sample,
)
from teach.data.tools import lengths_to_mask
import random
from pytorch3d.transforms import axis_angle_to_quaternion, quaternion_to_axis_angle
from data_loaders.d2m.quaternion import ax_from_6v, quat_slerp

torch.multiprocessing.set_sharing_strategy("file_system")


class GenerateDataset(Dataset):
    def __init__(
        self,
        data_path: str,
        feature_type: str = "baseline",
        normalizer: Any = None,
    ):
        self.data_path = data_path
        # print(self.data_path)

        self.feature_type = feature_type

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
        feature = torch.from_numpy(np.load(filename_))
        num_feats, feature_slice = slice_audio(filename_, 3.0)
        feature_slice = torch.from_numpy(feature_slice)

        data_length = 90
        length_transistion = 60
        
        # print("FEATURE SHAPE: ", feature_slice.shape)
        return {
            "length": data_length,
            "music": feature_slice,
            "filename": filename_
        }

    def load_data(self):
        # open data path
        sound_path = os.path.join(self.data_path, "feature")
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
    while start_idx < len(audio):
        if start_idx + window <= len(audio):
            audio_slice = audio[start_idx : start_idx + window]
        else:
            missing_length = window - (len(audio) - start_idx)
            missing_audio_slice = np.zeros((missing_length, 35))
            audio_slice = np.concatenate((audio[start_idx:], missing_audio_slice))

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


def get_dataset(args):
    DATA = GenerateDataset

    step = parse_resume_step_from_filename(args.model_path)

    normalizer_checkpoint = bf.join(
        bf.dirname(args.model_path), f"normalizer-{step:09}.pt"
    )

    checkpoint = torch.load(normalizer_checkpoint)
    loaded_normalizer = checkpoint["normalizer"]

    dataset = DATA(data_path=os.path.join(args.music_dir), normalizer=loaded_normalizer)

    return dataset


def get_dataset_loader(args, batch_size):
    dataset = get_dataset(args)
    num_cpus = multiprocessing.cpu_count()

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
        drop_last=True,
        # collate_fn=collate
    )

    return loader, dataset.normalizer


def create_model_kwargs(batch_index):
    model_kwargs_0 = {}
    model_kwargs_0["y"] = {}
    if args.inter_frames > 0:
        model_kwargs_0["y"]["lengths"] = [
            len + args.inter_frames // 2 for len in batch["length"]
        ]
    else:
        model_kwargs_0["y"]["lengths"] = batch["length"]
    model_kwargs_0["y"]["music"] = batch["music"][0][batch_index - 1].to(device)
    # print(batch["music"])
    model_kwargs_0["y"]["mask"] = (
        lengths_to_mask(model_kwargs_0["y"]["lengths"], dist_util.dev())
        .unsqueeze(1)
        .unsqueeze(2)
    )

    model_kwargs_1 = {}
    model_kwargs_1["y"] = {}

    if args.inter_frames > 0:
        model_kwargs_1["y"]["lengths"] = [
            len + args.inter_frames // 2 for len in batch["length"]
        ]
    else:
        model_kwargs_1["y"]["lengths"] = [
            args.inpainting_frames + len for len in batch["length"]
        ]
    model_kwargs_1["y"]["music"] = batch["music"][0][batch_index].to(device)
    model_kwargs_1["y"]["mask"] = (
        lengths_to_mask(model_kwargs_1["y"]["lengths"], dist_util.dev())
        .unsqueeze(1)
        .unsqueeze(2)
    )
    with open(f"./output_preview/model_kwargs/org_{i}.txt", "w") as file:
        file.write(f"model 0:\n{model_kwargs_0}\n")
        file.write(f"model 1:\n{model_kwargs_1}\n")

    return model_kwargs_0, model_kwargs_1


if __name__ == "__main__":
    args = generate_args()
    fixseed(args.seed)
    # TODO: fix the hardcode
    music_dir_len = len(os.listdir(args.music_dir))
    if music_dir_len > 32:
        args.batch_size = 32  # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    else:
        args.batch_size = 1
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
        log_file += f'_refine{args.refine_scale}'
    log_file += f'_comp{args.inter_frames}'
    log_file += '.log'
    # print(f'Will save to log file [{log_file}]')
    
    ########################################################################
    # LOAD SMPL

    smpl = SMPLSkeleton(device="cpu")

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
    baseline_extract(args.music_dir, dest=os.path.join(args.music_dir, "feature"))

    logger.log("creating data loader...")
    split = False

    #!: mỗi batch = 1 bài hát. vd có 2 bài => origin loader gồm 2 batch, mỗi batch sẽ chứa feature được sliced ra từ bài hát load vào
    dataloader, _ = get_dataset_loader(args, batch_size=args.batch_size)

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

    #! generate ở đây nè bùm bùm

    generated_motion = []
    mm_generated_motions = []
    clip_denoised = False  #! hardcoded (from repo)
    if args.refine:
        sample_fn_refine = diffusion.p_sample_loop

    wav_dir = "custom_input/feature"
    wavs = sorted(glob.glob(f"{wav_dir}/*.npy"))
    wav_out = wav_dir + "_sliced"

    with torch.no_grad():
        for i, batch in tqdm(enumerate(dataloader)):
            if (
                num_samples_limit is not None
                and len(generated_motion) >= num_samples_limit
            ):
                break

            with open("./output.txt", "w") as file:
                file.write(f"batch:\n{batch}\n")
                batch_music = batch["music"]
                file.write(f"{i}")
                file.write(f"\nmusic:\n{batch_music}\n")
                file.write(f"\nmusic length:{len(batch_music[0])}\n")

            n_slices = len(batch_music[0])

            for i in range(1, n_slices):
                pre_index, pos_index = i - 1, i
                with open(f"./output_preview/output_{i}.txt", "w") as file:
                    file.write(f"pre:\n{batch_music[0][i-1]}\n\n")
                    file.write(f"post:\n{batch_music[0][i]}\n")
                    file.write(f"squeeze:\n{batch_music.squeeze()}")

                # if args.inter_frames > 0:
                #     assert args.inter_frames % 2 == 0
                #     batch['length_0'] = [len + args.inter_frames // 2 for len in batch['length_0']]
                #     batch['length_1_with_transition'] = [len + args.inter_frames // 2 for len in batch['length_1_with_transition']]
                bs = len(batch["length"])
                model_kwargs_0, model_kwargs_1 = create_model_kwargs(i)
                with open(f"./output_preview/model_kwargs/fn_{i}.txt", "w") as file:
                    file.write(f"model 0:\n{model_kwargs_0}\n")
                    file.write(f"model 1:\n{model_kwargs_1}\n")

                model_kwargs_0 = {}
                model_kwargs_0["y"] = {}
                if args.inter_frames > 0:
                    model_kwargs_0["y"]["lengths"] = [
                        len + args.inter_frames // 2 for len in batch["length"]
                    ]
                else:
                    model_kwargs_0["y"]["lengths"] = batch["length"]
                model_kwargs_0["y"]["music"] = batch["music"].to(device)
                print(batch["music"])
                model_kwargs_0["y"]["mask"] = (
                    lengths_to_mask(model_kwargs_0["y"]["lengths"], dist_util.dev())
                    .unsqueeze(1)
                    .unsqueeze(2)
                )

                model_kwargs_1 = {}
                model_kwargs_1["y"] = {}

                if args.inter_frames > 0:
                    model_kwargs_1["y"]["lengths"] = [
                        len + args.inter_frames // 2 for len in batch["length"]
                    ]
                else:
                    model_kwargs_1["y"]["lengths"] = [
                        args.inpainting_frames + len for len in batch["length"]
                    ]
                model_kwargs_1["y"]["music"] = batch["music"].to(device)
                model_kwargs_1["y"]["mask"] = (
                    lengths_to_mask(model_kwargs_1["y"]["lengths"], dist_util.dev())
                    .unsqueeze(1)
                    .unsqueeze(2)
                )
                with open(f"./output_preview/model_kwargs/org_{i}.txt", "w") as file:
                    file.write(f"model 0:\n{model_kwargs_0}\n")
                    file.write(f"model 1:\n{model_kwargs_1}\n")

            break
            # add CFG scale to batch
            if args.guidance_param != 1.0:
                model_kwargs_0["y"]["scale"] = (
                    torch.ones(
                        len(model_kwargs_0["y"]["lengths"]),
                        device="cuda:0" if torch.cuda.is_available() else "cpu",
                    )
                    * args.guidedance_param
                )
                model_kwargs_1["y"]["scale"] = (
                    torch.ones(
                        len(model_kwargs_1["y"]["lengths"]),
                        device="cuda:0" if torch.cuda.is_available() else "cpu",
                    )
                    * args.guidedance_param
                )

            if args.shuffle_noise:
                feats = 151
                nframe = model_kwargs_0["y"]["mask"].shape[-1]

                noise_frame = 10
                noise_stride = 5

                noise_0 = torch.randn([1, feats, 1, nframe], device=device).repeat(
                    bs, 1, 1, 1
                )
                for frame_index in range(noise_frame, nframe, noise_stride):
                    list_index = list(
                        range(
                            frame_index - noise_frame,
                            frame_index + noise_stride - noise_frame,
                        )
                    )
                    random.shuffle(list_index)
                    noise_0[:, :, :, frame_index : frame_index + noise_stride] = (
                        noise_0[:, :, :, list_index]
                    )

                nframe = model_kwargs_1["y"]["mask"].shape[-1]

                noise_1 = torch.randn([1, feats, 1, nframe], device=device).repeat(
                    bs, 1, 1, 1
                )
                for frame_index in range(noise_frame, nframe, noise_stride):
                    list_index = list(
                        range(
                            frame_index - noise_frame,
                            frame_index + noise_stride - noise_frame,
                        )
                    )
                    random.shuffle(list_index)
                    noise_1[:, :, :, frame_index : frame_index + noise_stride] = (
                        noise_1[:, :, :, list_index]
                    )
            else:
                noise_1, noise_0 = None, None

            mm_num_now = len(mm_generated_motions) // dataloader.batch_size
            is_mm = False
            repeat_times = mm_num_repeats if is_mm else 1
            mm_motions = []

            for t in range(repeat_times):
                shape_0 = (bs, 151, 1, model_kwargs_0["y"]["mask"].shape[-1])
                shape_1 = (bs, 151, 1, model_kwargs_1["y"]["mask"].shape[-1])

                dump, sample_0 = create_pre_sample(
                    model,
                    shape_0,
                    noise_0=noise_0,
                    clip_denoised=clip_denoised,
                    model_kwargs_0=model_kwargs_0,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    const_noise=False,
                )

                dump, sample_1 = create_post_sample(
                    dump,
                    sample_0,
                    model,
                    args.hist_frames,
                    args.inpainting_frames,
                    shape_0,
                    shape_1,
                    noise_1=noise_1,
                    clip_denoised=clip_denoised,
                    model_kwargs_0=model_kwargs_0,
                    model_kwargs_1=model_kwargs_1,
                    skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                    init_image=None,
                    progress=False,
                    dump_steps=None,
                    const_noise=False,
                )
                # sample_0, sample_1 = sample_fn(
                #     model,
                #     args.hist_frames,
                #     args.inpainting_frames if not args.composition else args.inter_frames,
                #     (bs, 151, 1, model_kwargs_0['y']['mask'].shape[-1]),
                #     (bs, 151, 1, model_kwargs_1['y']['mask'].shape[-1]),
                #     noise_0=None,
                #     noise_1=None,
                #     clip_denoised=clip_denoised,
                #     model_kwargs_0=model_kwargs_0,
                #     model_kwargs_1=model_kwargs_1,
                #     skip_timesteps=0,  # 0 is the default value - i.e. don't skip any step
                #     init_image=None,
                #     progress=False,
                #     dump_steps=None,
                #     const_noise=False,
                #     # when experimenting guidance_scale we want to nutrileze the effect of noise on generation
                # )

                if args.inpainting_frames > 0:
                    sample_1_tmp = sample_1[
                        :, :, :, args.inpainting_frames :
                    ]  # B 135 1 L
                if args.inter_frames > 0:
                    sample_0 = sample_0[:, :, :, : -args.inter_frames // 2]
                    sample_1 = sample_1[:, :, :, args.inter_frames // 2 :]
                if args.refine:
                    model_kwargs_0_refine = {}
                    model_kwargs_0_refine['y'] = {}
                    if scale != 1.:
                        model_kwargs_0_refine['y']['scale'] = model_kwargs_0['y']['scale']
                    model_kwargs_0_refine['y']['music'] = model_kwargs_0['y']['music']
                    model_kwargs_0_refine['y']['next_motion'] = sample_1_tmp[:,:,:,:args.inpainting_frames]
                    model_kwargs_0_refine['y']['lengths'] = [len + args.inpainting_frames
                                                    for len in model_kwargs_0['y']['lengths']]
                    model_kwargs_0_refine['y']['mask'] = lengths_to_mask(model_kwargs_0_refine['y']['lengths'], dist_util.dev()).unsqueeze(1).unsqueeze(2)
                    
                    sample_0_refine = sample_fn_refine( # bs 135 1 len+inpainting 
                                                model,
                                                (bs, 151, 1, model_kwargs_0_refine['y']['mask'].shape[-1]),
                                                noise=None,
                                                clip_denoised=False,
                                                model_kwargs=model_kwargs_0_refine,
                                                skip_timesteps=0, 
                                                init_image=None,
                                                progress=True,
                                                dump_steps=None,
                                                const_noise=False)
                    # print("CHECKING: ", sample_0_refine.shape, sample_1.shape)
                    assert sample_0_refine.shape == sample_1.shape == (bs, 151, 1, 120)

                    sample_0_refine = sample_0_refine[
                        :, :, :, : -args.inpainting_frames
                    ]
                    to_stack = sample_0_refine[:, :, :, -args.inpainting_frames :]
                    sample_0 = sample_0 + args.refine_scale * (
                        sample_0_refine - sample_0
                    )

                    sample_0 = torch.cat((sample_0, to_stack), axis=-1)

                    # print(sample_0.shape, sample_1.shape)
                    assert sample_0.shape == sample_1.shape == (bs, 151, 1, 120)

                    sample = []

                    for idx in range(bs):
                        motion_0_result = (
                            sample_0[idx].squeeze().unsqueeze(dim=0).permute(0, 2, 1)
                        )
                        motion_1_result = (
                            sample_1[idx].squeeze().unsqueeze(dim=0).permute(0, 2, 1)
                        )

                        assert (
                            motion_0_result.shape
                            == motion_1_result.shape
                            == (1, 120, 151)
                        )

                        motion_result = torch.cat(
                            (motion_0_result, motion_1_result), dim=0
                        )

                        # print(motion_result.shape)

                        assert motion_result.shape == (2, 120, 151)

                        if motion_result.shape[2] == 151:
                            sample_contact, motion_result = torch.split(
                                motion_result, (4, motion_result.shape[2] - 4), dim=2
                            )
                        else:
                            sample_contact = None
                        # do the FK all at once
                        b, s, c = motion_result.shape
                        pos = motion_result[:, :, :3].to(
                            device
                        )  # np.zeros((sample.shape[0], 3))
                        q = motion_result[:, :, 3:].reshape(b, s, 24, 6)
                        # go 6d to ax
                        q = ax_from_6v(q).to(device)

                        b, s, c1, c2 = q.shape
                        assert s % 2 == 0
                        half = s // 2
                        if b > 1:
                            # if long mode, stitch position using linear interp

                            fade_out = torch.ones((1, s, 1)).to(pos.device)
                            fade_in = torch.ones((1, s, 1)).to(pos.device)
                            fade_out[:, half:, :] = torch.linspace(1, 0, half)[
                                None, :, None
                            ].to(pos.device)
                            fade_in[:, :half, :] = torch.linspace(0, 1, half)[
                                None, :, None
                            ].to(pos.device)

                            pos[:-1] *= fade_out
                            pos[1:] *= fade_in

                            full_pos = torch.zeros((s + half * (b - 1), 3)).to(
                                pos.device
                            )
                            id_ = 0
                            for pos_slice in pos:
                                full_pos[id_ : id_ + s] += pos_slice
                                id_ += half

                            # stitch joint angles with slerp
                            slerp_weight = torch.linspace(0, 1, half)[None, :, None].to(
                                pos.device
                            )

                            left, right = q[:-1, half:], q[1:, :half]
                            # convert to quat
                            left, right = (
                                axis_angle_to_quaternion(left),
                                axis_angle_to_quaternion(right),
                            )
                            merged = quat_slerp(
                                left, right, slerp_weight
                            )  # (b-1) x half x ...
                            # convert back
                            merged = quaternion_to_axis_angle(merged)

                            full_q = torch.zeros((s + half * (b - 1), c1, c2)).to(
                                pos.device
                            )
                            full_q[:half] += q[0, :half]
                            id_ = half
                            for q_slice in merged:
                                full_q[id_ : id_ + half] += q_slice
                                id_ += half
                            full_q[id_ : id_ + half] += q[-1, half:]

                            # unsqueeze for fk
                            full_pos = full_pos.unsqueeze(0)
                            full_q = full_q.unsqueeze(0)

                            full_pose = (
                                smpl.forward(full_q, full_pos).detach().cpu().numpy()
                            )  # b, s, 24, 3

                            # assert full_pose.shape == (1, 180, 24, 3)

                            filename = batch["filename"][idx]
                            outname = f'evaluation/inference_edge/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
                            out_path = os.path.join("./", outname)
                            # Create the directory if it doesn't exist
                            os.makedirs(os.path.dirname(out_path), exist_ok=True)
                            # print(out_path)
                            with open(out_path, "wb") as file_pickle:
                                pickle.dump(
                                    {
                                        "smpl_poses": full_q.squeeze(0)
                                        .reshape((-1, 72))
                                        .cpu()
                                        .numpy(),
                                        "smpl_trans": full_pos.squeeze(0).cpu().numpy(),
                                        "full_pose": full_pose.squeeze(),
                                    },
                                    file_pickle,
                                )

                            sample.append(full_pose)
