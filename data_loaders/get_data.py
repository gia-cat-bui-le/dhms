from torch.utils.data import DataLoader
from data_loaders.tensors import collate as all_collate
from data_loaders.tensors import t2m_collate
from teach.data.tools.collate import collate_pairs_and_text, collate_datastruct_and_text, collate_contrastive
from tqdm import tqdm 
from teach.data.sampling.base import FrameSampler

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

def get_collate_fn(split):
    return all_collate

def get_dataset(args, name, split=True):
    DATA = get_dataset_class(name)
    dataset = DATA(
        data_path=args.data_dir,
        train=split,
        force_reload=args.force_reload,
    )
    return dataset

def get_dataset_loader(args, name, batch_size, split=True):
    dataset = get_dataset(args, name, split)
    num_cpus = multiprocessing.cpu_count()
    
    collate = get_collate_fn(split)
    
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=min(int(num_cpus * 0.75), 32),
        pin_memory=True,
        drop_last=True,
        collate_fn=collate
    )
    
    return loader
