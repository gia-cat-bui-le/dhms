import glob
import os
import pickle

import librosa as lr
import numpy as np
import soundfile as sf
from tqdm import tqdm
import math
import numpy as np

def slice_audio(audio_file, stride, length, out_dir, num_slices, inpainting_frames, motion_len):
    # stride, length in seconds
    FPS = 30
    audio = np.load(audio_file)
    # print("audio: ", len(audio))
    file_name = os.path.splitext(os.path.basename(audio_file))[0]
    start_idx = 0
    idx = 0
    window = int(length * FPS)
    stride_step = int(stride * FPS)
    inpainting_frames_ = int(inpainting_frames * FPS)
    motion_len_ = int(motion_len * FPS)
    while start_idx <= len(audio) - window and idx < num_slices:
        audio_slice = audio[start_idx : start_idx + window]
        # audio_slice = [audio[start_idx : start_idx + motion_len_], audio[start_idx : start_idx + motion_len_ + inpainting_frames_], audio[start_idx + motion_len_ : start_idx + window]]
        # sf.write(f"{out_dir}/{file_name}_slice{idx}.wav", audio_slice, sr)
        np.save(f"{out_dir}/{file_name}_slice{idx}.npy", audio_slice)
        start_idx += stride_step
        idx += 1
    return idx

def slice_motion_finedance(motion_file, stride, length, out_dir, inpainting_frames, motion_len):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 30)
    stride_step = int(stride * 30)
    inpainting_frames_ = int(inpainting_frames * 30)
    motion_len_ = int(motion_len * 30)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window:
        pos_0, q_0 = (
            pos[start_idx : start_idx + motion_len_],
            q[start_idx : start_idx + motion_len_],
        ) # lenght_0
       
        pos_1, q_1 = (
            pos[start_idx + motion_len_ : start_idx + window],
            q[start_idx + motion_len_ : start_idx + window],
        ) # lenght_0
                    
        out = {"pos_0": pos_0, 
                "q_0":q_0,
                "pos_1": pos_1,
                "q_1": q_1,
                "length_0": int(motion_len_),
                "length_1": int(motion_len_),
                "length_transition": int(inpainting_frames_),
                }
        
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count

def slice_motion(motion_file, stride, length, out_dir, inpainting_frames, motion_len):
    motion = pickle.load(open(motion_file, "rb"))
    pos, q = motion["pos"], motion["q"]
    scale = motion["scale"][0]

    file_name = os.path.splitext(os.path.basename(motion_file))[0]
    # normalize root position
    pos /= scale
    start_idx = 0
    window = int(length * 60)
    stride_step = int(stride * 60)
    inpainting_frames_ = int(inpainting_frames * 60)
    motion_len_ = int(motion_len * 60)
    slice_count = 0
    # slice until done or until matching audio slices
    while start_idx <= len(pos) - window:
        
        pos_0, q_0 = (
            pos[start_idx : start_idx + motion_len_],
            q[start_idx : start_idx + motion_len_],
        ) # lenght_0
        pos_1, q_1 = (
            pos[start_idx + motion_len_ : start_idx + window],
            q[start_idx + motion_len_ : start_idx + window],
        ) # lenght_0
                    
        out = {"pos_0": pos_0, 
                "q_0":q_0,
                "pos_1": pos_1,
                "q_1": q_1,
                "length_0": int(motion_len_ / 2),
                "length_1": int(motion_len_ / 2),
                "length_transition": int(inpainting_frames_ / 2),
                }
        
        # out = {"pos": pos_slice, "q": q_slice, "pos1": pos_slice_1, "q1": q_slice_1}
        pickle.dump(out, open(f"{out_dir}/{file_name}_slice{slice_count}.pkl", "wb"))
        start_idx += stride_step
        slice_count += 1
    return slice_count


def slice_aistpp(motion_dir, wav_dir, stride=0.5, length=5, inpainting_frames=2.5, motion_len=2.5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.npy"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        motion_slices = slice_motion(motion, stride, length, motion_out, inpainting_frames, motion_len)
        audio_slices = slice_audio(wav, stride, length, wav_out, motion_slices, inpainting_frames, motion_len)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )
        
def slice_finedance(motion_dir, wav_dir, stride=0.5, length=5, inpainting_frames=2.5, motion_len=2.5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.npy"))
    motions = sorted(glob.glob(f"{motion_dir}/*.pkl"))
    wav_out = wav_dir + "_sliced"
    motion_out = motion_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    os.makedirs(motion_out, exist_ok=True)
    assert len(wavs) == len(motions)
    for wav, motion in tqdm(zip(wavs, motions)):
        # make sure name is matching
        music_fea = np.load(wav)
        motion_fea = pickle.load(open(motion, "rb"))
        pos, q, scale = motion_fea["pos"], motion_fea["q"], motion_fea["scale"]
        max_length = min(music_fea.shape[0], q.shape[0])

        music_fea = music_fea[:max_length, :]
        pos = pos[:max_length, :]
        q = q[:max_length, :]
        out_data = {"pos": pos, "q": q, "scale": scale}
        
        pickle.dump(out_data, open(motion, "wb"))
        np.save(wav, music_fea)
        
        m_name = os.path.splitext(os.path.basename(motion))[0]
        w_name = os.path.splitext(os.path.basename(wav))[0]
        assert m_name == w_name, str((motion, wav))
        motion_slices = slice_motion_finedance(motion, stride, length, motion_out, inpainting_frames, motion_len)
        audio_slices = slice_audio(wav, stride, length, wav_out, motion_slices, inpainting_frames, motion_len)
        # make sure the slices line up
        assert audio_slices == motion_slices, str(
            (wav, motion, audio_slices, motion_slices)
        )


def slice_audio_folder(wav_dir, stride=0.5, length=5):
    wavs = sorted(glob.glob(f"{wav_dir}/*.wav"))
    wav_out = wav_dir + "_sliced"
    os.makedirs(wav_out, exist_ok=True)
    for wav in tqdm(wavs):
        audio_slices = slice_audio(wav, stride, length, wav_out)