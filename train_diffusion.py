# This code is based on https://github.com/openai/guided-diffusion
"""
Train a diffusion model on images.
"""

import os
import json
from utils.fixseed import fixseed
from utils.parser_util import train_args
from utils import dist_util
from train.training_loop import TrainLoop
from data_loaders.get_data import get_dataset_loader
from utils.model_util import create_model_and_diffusion

def main():
    args = train_args()
    fixseed(args.seed)

    if args.save_dir is None:
        raise FileNotFoundError('save_dir was not specified.')
    elif os.path.exists(args.save_dir) and not args.overwrite:
        raise FileExistsError('save_dir [{}] already exists.'.format(args.save_dir))
    elif not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    args_path = os.path.join(args.save_dir, 'args.json')
    with open(args_path, 'w') as fw:
        json.dump(vars(args), fw, indent=4, sort_keys=True)
        
    if args.dataset == "aistpp":
        args.data_dir = os.path.join(args.data_dir, "aistpp_dataset")
    elif args.dataset == "finedance":
        args.data_dir = os.path.join(args.data_dir, "finedance")

    dist_util.setup_dist(args.device)

    print("creating data loader...")
    #TODO: check if we need num_frames in dataloader (cut each sequence into a fixed number of frames.
    # params can be set at parser_util.py
    data = get_dataset_loader(args, name=args.dataset, batch_size=args.batch_size, split=True)
    
    import numpy as np 

    print("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(args, data)
    model.to(dist_util.dev())
    # model.rot2xyz.smpl_model.eval()

    # print('Total params: %.2fM' % (sum(p.numel() for p in model.parameters_wo_clip()) / 1000000.0))
    print("Training...")
    loop = TrainLoop(args, model, diffusion, data)
    loop.run_loop()

if __name__ == "__main__":
    main()
