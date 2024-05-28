import numpy as np
import pickle 
from evaluation.features.kinetic import extract_kinetic_features
from evaluation.features.manual_new import extract_manual_features
from scipy import linalg
import json
# kinetic, manual
import os
from  scipy.ndimage import gaussian_filter as G
from scipy.signal import argrelextrema
import librosa
import matplotlib.pyplot as plt 

music_root = 'result\FACT\wav'


def get_mb(key, length=None):
    #TODO: fix the key in generated motion
    path = os.path.join(music_root, key)
    with open(path) as f:
        #print(path)
        sample_dict = json.loads(f.read())
        if length is not None:
            beats = np.array(sample_dict['music_array'])[:, 53][:][:length]
        else:
            beats = np.array(sample_dict['music_array'])[:, 53]


        beats = beats.astype(bool)
        beat_axis = np.arange(len(beats))
        beat_axis = beat_axis[beats]
        
        # fig, ax = plt.subplots()
        # ax.set_xticks(beat_axis, minor=True)
        # # ax.set_xticks([0.3, 0.55, 0.7], minor=True)
        # ax.xaxis.grid(color='deeppink', linestyle='--', linewidth=1.5, which='minor')
        # ax.xaxis.grid(True, which='minor')


        # print(len(beats))
        return beat_axis

def get_music_beat_fromwav(fpath, length):
    FPS = 30
    HOP_LENGTH = 512
    SR = FPS * HOP_LENGTH
    # EPS = 1e-6
    data, _ = librosa.load(fpath, sr=SR)[:length]
    # print("loaded music data shape", data.shape)
    envelope = librosa.onset.onset_strength(y=data, sr=SR)  # (seq_len,)
    peak_idxs = librosa.onset.onset_detect(
        onset_envelope=envelope.flatten(), sr=SR, hop_length=HOP_LENGTH
    )
    start_bpm = librosa.beat.tempo(y=data)[0]
    tempo, beat_idxs = librosa.beat.beat_track(
        onset_envelope=envelope,
        sr=SR,
        hop_length=HOP_LENGTH,
        start_bpm=start_bpm,
        tightness=100,
    )
    return beat_idxs

def get_music_beat_from_finedance(fpath, length):
    data = np.load(fpath)[:length]
    beat_idxs = data[-1]

    beats = beats.astype(bool)
    beat_axis = np.arange(len(beats))
    beat_axis = beat_axis[beats]
  
    return beat_idxs

from evaluation.metrics_new import *
from evaluation.beat_align import *
from evaluation.pfc import *
from evaluation.metrics_new import quantized_metrics, calc_and_save_feats
import torch 
from scipy.spatial.transform import Rotation as R
from evaluation.features.kinetic import extract_kinetic_features
from evaluation.features.manual_new import extract_manual_features
from vis import SMPLSkeleton

def eye(n, batch_shape):
    iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
    iden[..., 0, 0] = 1.0
    iden[..., 1, 1] = 1.0
    iden[..., 2, 2] = 1.0
    return iden

def get_closest_rotmat(rotmats):
    """
    Finds the rotation matrix that is closest to the inputs in terms of the Frobenius norm. For each input matrix
    it computes the SVD as R = USV' and sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.
    Args:
        rotmats: np array of shape (..., 3, 3).
    Returns:
        A numpy array of the same shape as the inputs.
    """
    u, s, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    # if the determinant of UV' is -1, we must flip the sign of the last column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest

def recover_to_axis_angles(motion):
    batch_size, seq_len, dim = motion.shape
    assert dim == 225
    transl = motion[:, :, 6:9]
    rotmats = get_closest_rotmat(
        np.reshape(motion[:, :, 9:], (batch_size, seq_len, 24, 3, 3))
    )
    axis_angles = R.from_matrix(
        rotmats.reshape(-1, 3, 3)
    ).as_rotvec().reshape(batch_size, seq_len, 24, 3)
    return axis_angles, transl

def extract_feature(motion, smpl_model, mode="kinetic"):
    smpl_poses, smpl_trans = recover_to_axis_angles(motion)
    smpl_poses = np.squeeze(smpl_poses, axis=0)  # (seq_len, 24, 3)
    smpl_trans = np.squeeze(smpl_trans, axis=0)  # (seq_len, 3)
    keypoints3d = smpl_model.forward(
        global_orient=torch.from_numpy(smpl_poses[:, 0:1]).float(),
        body_pose=torch.from_numpy(smpl_poses[:, 1:]).float(),
        transl=torch.from_numpy(smpl_trans).float(),
    ).joints.detach().numpy()[:, :24, :]   # (seq_len, 24, 3)

    return keypoints3d

def calc_db(keypoints, name=''):
    keypoints = np.array(keypoints).reshape(-1, 24, 3)
    kinetic_vel = np.mean(np.sqrt(np.sum((keypoints[1:] - keypoints[:-1]) ** 2, axis=2)), axis=1)
    kinetic_vel = G(kinetic_vel, 5)
    motion_beats = argrelextrema(kinetic_vel, np.less)
    return motion_beats, len(kinetic_vel)


def BA(music_beats, motion_beats):
    ba = 0
    for bb in music_beats:
        ba +=  np.exp(-np.min((motion_beats[0] - bb)**2) / 2 / 9)
    return (ba / len(music_beats))

from smplx import SMPL

def calc_ba_score(root):
    
    smpl = SMPL(model_path="result\FACT\smpl", gender='MALE', batch_size=1)

    # gt_list = []
    ba_scores = []

    for pkl in tqdm(os.listdir(root)):
        # print(pkl)
        if os.path.isdir(os.path.join(root, pkl)):
            continue
        result_motion = np.load(open(os.path.isdir(os.path.join(root, pkl)), "rb"), allow_pickle=True)[None, ...]
        joint3d = extract_feature(result_motion[:, 120:], smpl, "kinetic")

        dance_beats, length = calc_db(joint3d, pkl)        
        music_beats = get_music_beat_fromwav(os.path.join(music_root, pkl.split('.')[0][-4:] + '.wav'), joint3d.shape[0])

        ba_scores.append(BA(music_beats, dance_beats))
        
    return np.mean(ba_scores)

if __name__ == '__main__':

    # aa = np.random.randn(39, 72)*
    # bb = np.random.randn(39, 72)*0.1
    # print(calc_fid(aa, bb))
    # gt_root = '/mnt/lustre/lisiyao1/dance/bailando/aist_features_zero_start'
    # pred_root = '/mnt/lustressd/lisiyao1/dance_experiements/experiments/sep_vqvae_root_global_vel_wav_acc_batch8/vis/pkl/ep000500'
    # pred_root = ''
    pred_root = 'save\\results'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_gpt_ds8_lbin512_c512_di3full/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_3_9_9_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/eval/pkl/ep000020'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_3_9_9_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/vis/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_wav_bsz_16_layer6/eval/pkl/ep000040'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/sep_vqvae_root_data_l1_d8_local_c512_di3_global_vel_full_beta0.9_1e-4_wav_beta0.5/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_wav/eval/pkl/ep000300'
    # pred_root = '/mnt/lustre/lisiyao1/dance/bailando/experiments/music_cross_cond_gpt_ds8_lbin512_c512_di3_init_0.01_beta0.9_full_dim768_666_ac_reward2_with_entropy_loss_alpha0.5_lr1e-4_no_pretrain/vis/pkl/ep000080'
    # print('Calculating and saving features')
    print(calc_ba_score(pred_root))
    # calc_and_save_feats(gt_root)

    # print('Calculating metrics')
    # print(gt_root)
    # print(pred_root)
    # print(quantized_metrics(pred_root, gt_root))