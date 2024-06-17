import pickle
import logging
import numpy as np
import scipy
from scipy.spatial.transform import Rotation


def rotmat2euler(angles, seq="XYZ"):
    """Converts rotation matrices to axis angles.

    Args:
        angles: np array of shape [..., 3, 3] or [..., 9].
        seq: 3 characters belonging to the set {‘X’, ‘Y’, ‘Z’} for
        intrinsic rotations, or {‘x’, ‘y’, ‘z’} for extrinsic
        rotations. Used by `scipy.spatial.transform.Rotation.as_euler`.

    Returns:
        np array of shape [..., 3].
    """
    input_shape = angles.shape
    assert input_shape[-2:] == (3, 3) or input_shape[-1] == 9, (
        f"input shape is not valid! got {input_shape}")
    if input_shape[-2:] == (3, 3):
        output_shape = input_shape[:-2] + (3,)
    else:  # input_shape[-1] == 9
        output_shape = input_shape[:-1] + (3,)

    if scipy.__version__ < "1.4.0":
        # from_dcm is renamed to from_matrix in scipy 1.4.0 and will be
        # removed in scipy 1.6.0
        r = Rotation.from_dcm(angles.reshape(-1, 3, 3))
    else:
        r = Rotation.from_matrix(angles.reshape(-1, 3, 3))
    output = r.as_euler(seq, degrees=False).reshape(output_shape)
    return output


def rotmat2aa(angles):
    """Converts rotation matrices to axis angles.

    Args:
        angles: np array of shape [..., 3, 3] or [..., 9].

    Returns:
        np array of shape [..., 3].
    """
    input_shape = angles.shape
    assert input_shape[-2:] == (3, 3) or input_shape[-1] == 9, (
        f"input shape is not valid! got {input_shape}")
    if input_shape[-2:] == (3, 3):
        output_shape = input_shape[:-2] + (3,)
    else:  # input_shape[-1] == 9
        output_shape = input_shape[:-1] + (3,)

    if scipy.__version__ < "1.4.0":
        # from_dcm is renamed to from_matrix in scipy 1.4.0 and will be
        # removed in scipy 1.6.0
        r = Rotation.from_dcm(angles.reshape(-1, 3, 3))
    else:
        r = Rotation.from_matrix(angles.reshape(-1, 3, 3))
    output = r.as_rotvec().reshape(output_shape)
    return output


def aa2rotmat(angles):
    """Converts axis angles to rotation matrices.

    Args:
        angles: np array of shape [..., 3].

    Returns:
        np array of shape [..., 9].
    """
    input_shape = angles.shape
    assert input_shape[-1] == 3, (f"input shape is not valid! got {input_shape}")
    output_shape = input_shape[:-1] + (9,)

    r = Rotation.from_rotvec(angles.reshape(-1, 3))
    if scipy.__version__ < "1.4.0":
        # as_dcm is renamed to as_matrix in scipy 1.4.0 and will be
        # removed in scipy 1.6.0
        output = r.as_dcm().reshape(output_shape)
    else:
        output = r.as_matrix().reshape(output_shape)
    return output


def get_closest_rotmat(rotmats):
    """Compute the closest valid rotmat.

    Finds the rotation matrix that is closest to the inputs in terms of the
    Frobenius norm. For each input matrix it computes the SVD as R = USV' and
    sets R_closest = UV'. Additionally, it is made sure that det(R_closest) == 1.

    Args:
        rotmats: np array of shape (..., 3, 3) or (..., 9).

    Returns:
        A numpy array of the same shape as the inputs.
    """
    input_shape = rotmats.shape
    assert input_shape[-2:] == (3, 3) or input_shape[-1] == 9, (
        f"input shape is not valid! got {input_shape}")
    if input_shape[-1] == 9:
        rotmats = rotmats.reshape(input_shape[:-1] + (3, 3))

    u, _, vh = np.linalg.svd(rotmats)
    r_closest = np.matmul(u, vh)

    def _eye(n, batch_shape):
        iden = np.zeros(np.concatenate([batch_shape, [n, n]]))
        iden[..., 0, 0] = 1.0
        iden[..., 1, 1] = 1.0
        iden[..., 2, 2] = 1.0
        return iden

    # if the determinant of UV' is -1, we must flip the sign of the last
    # column of u
    det = np.linalg.det(r_closest)  # (..., )
    iden = _eye(3, det.shape)
    iden[..., 2, 2] = np.sign(det)
    r_closest = np.matmul(np.matmul(u, iden), vh)
    return r_closest.reshape(input_shape)

def _read_motion_from_pkl_file(pkl_filename):
    """Read motion from a pkl file."""
    try:
        with open(pkl_filename, "rb") as f:
            data = pickle.load(f)
    except EOFError as e:
        message = "Aboring reading file %s due to: %s" % (pkl_filename, str(e))
        logging.warning(message)
        raise ValueError(message)

    if "smpl_poses" in data:
        axis_angles = np.reshape(data["smpl_poses"], [-1, 24, 3])
        if data["smpl_trans"] is None:
            trans = np.zeros((axis_angles.shape[0], 3), dtype=np.float32)
        else:
            trans = np.reshape(data["smpl_trans"], [-1, 3])
    else:
        rotmats = np.reshape(data["pred_motion"], [-1, 24, 3, 3])
        rotmats = get_closest_rotmat(rotmats)
        axis_angles = rotmat2aa(rotmats)
        trans = np.reshape(data["pred_trans"], [-1, 3])
        
    return axis_angles, trans
        
if __name__ == "__main__":
    import glob
    import os
    import torch
    from vis import SMPLSkeleton
    
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    motion_dir = "result\Bailando\ep000010"
    
    for f in glob.glob(os.path.join(motion_dir, "*.npy")):
        smpl_poses, smpl_trans = _read_motion_from_pkl_file(f)
        smpl_poses = torch.Tensor(smpl_poses).unsqueeze(0)
        smpl_trans = torch.Tensor(smpl_trans).unsqueeze(0)
        
        smpl = SMPLSkeleton(device=device)
        
        full_pose = (
            smpl.forward(smpl_poses, smpl_trans).squeeze(0).detach().cpu().numpy()
        )  # b, s, 24, 3
        
        out_dir = os.path.join(motion_dir, "process", os.path.splitext(os.path.basename(f))[0])
        os.makedirs(os.path.dirname(out_dir), exist_ok=True)
        with open(out_dir, "wb") as file_pickle:
            pickle.dump(
                {
                    "smpl_poses": torch.Tensor(smpl_poses).squeeze(0).reshape((-1, 24 * 3)).cpu().numpy(),
                    "smpl_trans": smpl_trans.squeeze(0).cpu().numpy(),
                    "full_pose": full_pose.squeeze(),
                },
                file_pickle,
            )