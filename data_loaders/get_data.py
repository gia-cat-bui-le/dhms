from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import collate_pairs_and_text, collate_contrastive
# from data_loaders.tensors import t2m_collate
# from teach.data.tools.collate import collate_pairs_and_text, collate_datastruct_and_text, collate_contrastive
from tqdm import tqdm 
from teach.data.sampling.base import FrameSampler
import torch
import blobfile as bf

import multiprocessing

def get_dataset_class(name):
    if name == "aistpp":
        from data_loaders.d2m.dance_dataset import AISTPPDataset
        return AISTPPDataset
    elif name == "finedance":
        from data_loaders.d2m.dance_dataset import FineDanceDataset
        return FineDanceDataset
    else:
        raise ValueError(f'Unsupported dataset name [{name}]')

def get_collate_fn(split, hml_mode):
    if hml_mode == 'train':
        collate = collate_pairs_and_text
    else:
        collate = collate_contrastive
    return collate

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

def get_dataset(args, name, split=True):
    DATA = get_dataset_class(name)
    
    if split is False:
        
        # step = parse_resume_step_from_filename(args.model_path)
        
        # normalizer_checkpoint = bf.join(
        #     bf.dirname(args.model_path), f"normalizer-{step:09}.pt"
        # )
        
        # checkpoint = torch.load(normalizer_checkpoint)
        # loaded_normalizer = checkpoint["normalizer"]
        
        dataset = DATA(
        data_path=args.data_dir,
        train=split,
        force_reload=args.force_reload,
        normalizer=None
        # normalizer=loaded_normalizer
    )
    else:
        dataset = DATA(
            data_path=args.data_dir,
            train=split,
            force_reload=args.force_reload,
        )
    return dataset

def get_dataset_loader(args, name, batch_size, split=True, hml_mode='train'):
    dataset = get_dataset(args, name, split)
    num_cpus = multiprocessing.cpu_count()
    
    collate = get_collate_fn(split, hml_mode=hml_mode)
    
    if split:
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=min(int(num_cpus * 0.75), 32),
            pin_memory=True,
            drop_last=True,
            collate_fn=collate
        )
    else:
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
