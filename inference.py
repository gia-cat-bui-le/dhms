from utils.parser_util import evaluation_parser
from utils.fixseed import fixseed
from datetime import datetime
from data_loaders.humanml.motion_loaders.model_motion_loaders import get_mdm_loader, get_motion_loader
from data_loaders.humanml.utils.metrics import *
from data_loaders.humanml.networks.evaluator_wrapper import EvaluatorMDMWrapper, EvaluatorCCDWrapper
from collections import OrderedDict
from data_loaders.humanml.scripts.motion_process import *
from data_loaders.humanml.utils.utils import *
from utils.model_util import create_model_and_diffusion, load_model_wo_clip

from diffusion import logger
from utils import dist_util
from data_loaders.get_data import get_dataset_loader
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

torch.multiprocessing.set_sharing_strategy('file_system')

# def generating(motion_loaders, out_dir):
#     print('========== GENERATING ==========')
#     for motion_loader_name, motion_loader in motion_loaders.items():
#         all_motion_embedding = []
#         all_motion_embeddings = []
#         with torch.no_grad():
#             for idx, batch in enumerate(motion_loader):
#                 if motion_loader_name != 'vald':
#                     motion, length, music, filenames = batch["motion_feats"], batch["length"], batch["music"], batch["filename"]
#                 else:
#                     motion, length, music, filenames = batch
                    
#                 b, s, c = motion.shape
                
#                 sample_contact, motion = torch.split(
#                 motion, (4, motion.shape[2] - 4), dim=2)
#                 pos = motion[:, :, :3].to(motion.device)  # np.zeros((sample.shape[0], 3))
#                 q = motion[:, :, 3:].reshape(b, s, 24, 6)
#                 # go 6d to ax
#                 q = ax_from_6v(q).to(motion.device)
                
#                 full_poses = (smpl.forward(q, pos).detach().cpu().numpy())
                
#                 print(full_poses.shape)
                
#                 for full_pose, filename in zip(full_poses, filenames):
#                     print(full_pose.shape)
#                     if out_dir is not None:
#                         outname = f'evaluation/inference/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
#                         out_path = os.path.join(out_dir, outname)
#                         # Create the directory if it doesn't exist
#                         os.makedirs(os.path.dirname(out_path), exist_ok=True)
#                         # print(out_path)
#                         with open(out_path, "wb") as file_pickle:
#                             pickle.dump(
#                                 {
#                                     "smpl_poses": q.squeeze(0).reshape((-1, 72)).cpu().numpy(),
#                                     "smpl_trans": pos.squeeze(0).cpu().numpy(),
#                                     "full_pose": full_pose,
#                                 },
#                                 file_pickle,
#                             )
                
#                 #TODO: code save generated motion

    #             all_motion_embedding.append(full_poses)
                
    #         all_motion_embedding = np.concatenate(all_motion_embedding, axis=0)
    #     all_motion_embeddings.append(all_motion_embedding)
        
    # all_motion_embeddings = np.concatenate(all_motion_embeddings, axis=0)

def inference(eval_motion_loaders, origin_loader, out_dir, log_file, replication_times, diversity_times, mm_num_times, run_mm=False):
    with open(log_file, 'a') as f:
        for replication in range(replication_times):
            motion_loaders = {}
            for motion_loader_name, motion_loader_getter in eval_motion_loaders.items():
                motion_loader = motion_loader_getter()
                motion_loaders[motion_loader_name] = motion_loader

            print(f'==================== Replication {replication} ====================')
            print(f'==================== Replication {replication} ====================', file=f, flush=True)
            print(f'Time: {datetime.now()}')
            print(f'Time: {datetime.now()}', file=f, flush=True)
            
            # generating(motion_loaders, out_dir)
            
            for batch in origin_loader:
                motion_0, motion_1_with_transition, filenames = batch["motion_feats_0"], batch["motion_feats_1"], batch["filename"]
                motion = torch.from_numpy(np.concatenate((motion_0, motion_1_with_transition), axis=1))
                
                b, s, c = motion.shape
                
                sample_contact, motion = torch.split(
                motion, (4, motion.shape[2] - 4), dim=2)
                pos = motion[:, :, :3].to(motion.device)  # np.zeros((sample.shape[0], 3))
                q = motion[:, :, 3:].reshape(b, s, 24, 6)
                # go 6d to ax
                q = ax_from_6v(q).to(motion.device)
                
                full_poses = (smpl.forward(q, pos).detach().cpu().numpy())
                
                for full_pose, q_, pos_, filename in zip(full_poses, q, pos, filenames):
                    if out_dir is not None:
                        outname = f'evaluation/gt_edge/{"".join(os.path.splitext(os.path.basename(filename))[0])}.pkl'
                        out_path = os.path.join(out_dir, outname)
                        # Create the directory if it doesn't exist
                        os.makedirs(os.path.dirname(out_path), exist_ok=True)
                        print(out_path)
                        with open(out_path, "wb") as file_pickle:
                            pickle.dump(
                                {
                                    "smpl_poses": q_.squeeze(0).reshape((-1, 72)).cpu().numpy(),
                                    "smpl_trans": pos_.squeeze(0).cpu().numpy(),
                                    "full_pose": full_pose,
                                },
                                file_pickle,
                            )
                
                # print(batch["length_0"], batch["length_1"])

            print(f'!!! DONE !!!')
            print(f'!!! DONE !!!', file=f, flush=True)

if __name__ == '__main__':
    
    args = evaluation_parser()
    fixseed(args.seed)
    #TODO: fix the hardcode
    args.batch_size = 32 # This must be 32! Don't change it! otherwise it will cause a bug in R precision calc!
    name = os.path.basename(os.path.dirname(args.model_path))
    niter = os.path.basename(args.model_path).replace('model', '').replace('.pt', '')
    log_file = os.path.join(os.path.dirname(args.model_path), 'inference_{}_{}'.format(name, niter))
    
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
    
    # output data with shape [batch_size, n_frames, 24 joints, 3 dimension (each joint)]
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


    if args.dataset == "aistpp":
        args.data_dir = os.path.join(args.data_dir, "aistpp_dataset")
    elif args.dataset == "finedance":
        args.data_dir = os.path.join(args.data_dir, "finedance")

    dist_util.setup_dist(args.device)
    logger.configure()

    logger.log("creating data loader...")
    split = False
    origin_loader, _ = get_dataset_loader(args, name=args.dataset, batch_size=args.batch_size, split=split)
    
    # gt_loader = get_dataset_loader(name=args.dataset, batch_size=args.batch_size, split=split, hml_mode='eval')
    # num_actions = gen_loader.dataset.num_actions
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
        # generate data
        'vald': lambda: get_mdm_loader(
            args, model, diffusion, args.batch_size,
            origin_loader, mm_num_samples, mm_num_repeats, num_samples_limit, args.guidance_param
        )
        # 'vald': lambda: get_motion_loader(
        #     args, model, diffusion, args.batch_size,
        #     origin_loader, mm_num_samples, mm_num_repeats, num_samples_limit, args.guidance_param
        # )
    }

    # eval_wrapper = EvaluatorCCDWrapper(args.dataset, dist_util.dev())
    inference(eval_motion_loaders, origin_loader, args.out_dir, log_file, replication_times, diversity_times, mm_num_times, run_mm=run_mm)