from argparse import ArgumentParser
import argparse
import os
import json


def parse_and_load_from_model(parser):
    # args according to the loaded model
    # do not try to specify them from cmd line since they will be overwritten
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    args = parser.parse_args()
    args_to_overwrite = []
    for group_name in ['dataset', 'model', 'diffusion']:
        args_to_overwrite += get_args_per_group_name(parser, args, group_name)

    # load args from model
    model_path = get_model_path_from_args()
    args_path = os.path.join(os.path.dirname(model_path), 'args.json')
    assert os.path.exists(args_path), 'Arguments json file was not found!'
    with open(args_path, 'r') as fr:
        model_args = json.load(fr)

    for a in args_to_overwrite:
        if a in model_args.keys():
            setattr(args, a, model_args[a])

        elif 'cond_mode' in model_args: # backward compitability
            unconstrained = (model_args['cond_mode'] == 'no_cond')
            setattr(args, 'unconstrained', unconstrained)

        else:
            print('Warning: was not able to load [{}], using default value [{}] instead.'.format(a, args.__dict__[a]))

    if args.cond_drop_prob == 0:
        args.guidance_param = 1
    return args


def get_args_per_group_name(parser, args, group_name):
    for group in parser._action_groups:
        if group.title == group_name:
            group_dict = {a.dest: getattr(args, a.dest, None) for a in group._group_actions}
            return list(argparse.Namespace(**group_dict).__dict__.keys())
    return ValueError('group_name was not found.')

def get_model_path_from_args():
    try:
        dummy_parser = ArgumentParser()
        dummy_parser.add_argument('model_path')
        dummy_args, _ = dummy_parser.parse_known_args()
        print(dummy_args.model_path)
        return dummy_args.model_path
    except:
        raise ValueError('model_path argument must be specified.')


def add_base_options(parser):
    group = parser.add_argument_group('base')
    group.add_argument("--cuda", default=True, type=bool, help="Use cuda device, otherwise use CPU.")
    group.add_argument("--device", default=0, type=int, help="Device id to use.")
    group.add_argument("--seed", default=10, type=int, help="For fixing random seed.")
    group.add_argument("--batch_size", default=128, type=int, help="Batch size during training.")
    group.add_argument("--tiny", default=False, type=bool, help="If True, use only part of data")
    group.add_argument("--inpainting_frames", default=30, type=int, help="inpainting_frames")

def add_diffusion_options(parser):
    group = parser.add_argument_group('diffusion')
    group.add_argument("--noise_schedule", default='cosine', choices=['linear', 'cosine'], type=str,
                       help="Noise schedule type")
    group.add_argument("--diffusion_steps", default=1000, type=int,
                       help="Number of diffusion steps (denoted T in the paper)")
    group.add_argument("--sigma_small", default=True, type=bool, help="Use smaller sigma values.")


def add_model_options(parser):
    group = parser.add_argument_group('model')
    group.add_argument("--max_seq_att", default=1024, type=int,
                       help="Max window size for attention")
    group.add_argument("--layers", default=8, type=int,
                       help="Number of layers.")
    group.add_argument("--num_heads", default=4, type=int,
                       help="Number of heads per layer.")
    group.add_argument("--latent_dim", default=512, type=int,
                       help="Transformer/GRU width.")
    
    group.add_argument("--feature_type", default="baseline", type=str)
    group.add_argument("--cond_drop_prob", default=0.1, type=float,
                        help="The probability of masking the condition during training."
                            " For classifier-free guidance learning.")
    group.add_argument("--arch", default='inpainting',
                       choices=['trans_enc', 'trans_dec', 'gru', 'past_cond', 'inpainting'], type=str,
                       help="Architecture types as reported in the paper.")
    group.add_argument("--emb_trans_dec", default=False, type=bool,
                       help="For trans_dec architecture only, if true, will inject condition as a class token"
                            " (in addition to cross-attention).")
    group.add_argument("--lambda_mse", default=0.636, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_rcxyz", default=1.0, type=float, help="Joint positions loss.")
    group.add_argument("--lambda_vel", default=2.964, type=float, help="Joint velocity loss.")
    group.add_argument("--lambda_fc", default=10.942, type=float, help="Foot contact loss.")
    # group.add_argument("--unconstrained", action='store_true',
    #                    help="Model is trained unconditionally. That is, it is constrained by neither text nor action. "
    #                         "Currently tested on HumanAct12 only.")
    group.add_argument("--motion_mask", default=True, type=bool, help="if mask")
    
    
    group.add_argument(
        "--shuffle_noise",
        action="store_true",
        help="use noise shuffling or not",
    )

def add_data_options(parser):
    group = parser.add_argument_group('dataset')
    group.add_argument("--dataset", default='aistpp', choices=['aistpp', 'finedance'], type=str,
                       help="Dataset name (choose from list).")
    group.add_argument("--data_dir", default="/home/ltnghia02/data", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    group.add_argument("--inference_dir", default="/home/ltnghia02/data/evaluation", type=str,
                       help="If empty, will use defaults according to the specified dataset.")
    

def add_training_options(parser):
    group = parser.add_argument_group('training')
    group.add_argument("--save_dir", default='./result', required=True, type=str,
                       help="Path to save checkpoints and results.")
    group.add_argument("--overwrite", action='store_true',
                       help="If True, will enable to use an already existing save_dir.")
    group.add_argument("--lr", default=1e-4, type=float, help="Learning rate.")
    group.add_argument("--weight_decay", default=0, type=float, help="Optimizer weight decay.")
    group.add_argument("--lr_anneal_steps", default=200000, type=int, help="Number of learning rate anneal steps.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument("--eval_split", default='test', choices=['val', 'test'], type=str,
                       help="Which split to evaluate on during training.")
    group.add_argument("--eval_during_training", action='store_true',
                       help="If True, will run evaluation during training.")
    group.add_argument("--eval_rep_times", default=3, type=int,
                       help="Number of repetitions for evaluation loop during training.")
    group.add_argument("--eval_num_samples", default=1_000, type=int,
                       help="If -1, will use all samples in the specified split.")
    group.add_argument("--log_interval", default=1_000, type=int,
                       help="Log losses each N steps")
    group.add_argument("--save_interval", default=10_000, type=int,
                       help="Save checkpoints and run evaluation each N steps")
    group.add_argument("--num_steps", default=600_000, type=int,
                       help="Training will stop after the specified number of steps.")
    group.add_argument("--epochs", default=2000, type=int,
                       help="Training will stop after the specified number of epochs.")
    group.add_argument("--num_frames", default=60, type=int,
                       help="Limit for the maximal number of frames. In HumanML3D and KIT this field is ignored.")
    group.add_argument("--resume_checkpoint", default="", type=str,
                       help="If not empty, will start from the specified checkpoint (path to model###.pt file).")
    group.add_argument("--resume_step", action="store_true",)


def add_sampling_options(parser):
    group = parser.add_argument_group('sampling')
    group.add_argument("--model_path", required=True, type=str,
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--output_dir", default='', type=str,
                       help="Path to results dir (auto created by the script). "
                            "If empty, will create dir in parallel to checkpoint.")
    group.add_argument("--num_samples", default=0, type=int,
                       help="Maximal number of prompts to sample, "
                            "if loading dataset from file, this field will be ignored.")
    group.add_argument("--num_repetitions", default=0, type=int,
                       help="Number of repetitions, per sample (text prompt/action)")
    group.add_argument("--guidance_param", default=1.0, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")

#TODO: fix motion length
def add_generate_options(parser):
    group = parser.add_argument_group('generate')
    group.add_argument("--motion_length", default=5.0, type=float,
                       help="The length of the sampled motion [in seconds]. ")
    group.add_argument("--use_cached_features", action="store_true",
        help="Use precomputed features instead of music folder",
    )
    group.add_argument(
        "--feature_cache_dir",
        type=str,
        default="cached_features/",
        help="Where to save/load the features",
    )
    group.add_argument(
        "--music_dir",
        type=str,
        default="./custom_input_val",
        help="folder containing input music",
    )
    group.add_argument(
        "--cache_features",
        action="store_true",
        help="Save the jukebox features for later reuse",
    )

def add_evaluation_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", required=True, type=str, default='./',
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--eval_mode", default='debug', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument("--guidance_param", default=1.0, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--eval_batch_size", default=32, type=int,
                       help="Batch size during evaluation loop. Do not change this unless you know what you are doing. "
                            "T2m precision calculation is based on fixed batch size 32.")
    group.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="folder containing input music",
    )
    
def add_evaluation_during_training_options(parser):
    group = parser.add_argument_group('eval')
    group.add_argument("--model_path", type=str, default='./',
                       help="Path to model####.pt file to be sampled.")
    group.add_argument("--guidance_param", default=1.0, type=float,
                       help="For classifier-free sampling - specifies the s parameter, as defined in the paper.")
    group.add_argument("--eval_mode", default='debug', choices=['wo_mm', 'mm_short', 'debug', 'full'], type=str,
                       help="wo_mm (t2m only) - 20 repetitions without multi-modality metric; "
                            "mm_short (t2m only) - 5 repetitions with multi-modality metric; "
                            "debug - short run, less accurate results."
                            "full (a2m only) - 20 repetitions.")
    group.add_argument(
        "--out_dir",
        type=str,
        default="./",
        help="folder containing input music",
    )

def train_args():
    parser = ArgumentParser()
    add_base_options(parser)
    add_data_options(parser)
    add_model_options(parser)
    add_diffusion_options(parser)
    add_training_options(parser)
    add_evaluation_during_training_options(parser)
    return parser.parse_args()


def generate_args():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_sampling_options(parser)
    add_generate_options(parser)
    return parse_and_load_from_model(parser)

def evaluation_parser():
    parser = ArgumentParser()
    # args specified by the user: (all other will be loaded from the model)
    add_base_options(parser)
    add_evaluation_options(parser)
    return parse_and_load_from_model(parser)