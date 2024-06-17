import torch
import numpy as np
from typing import List, Dict

import warnings

# Suppress the warning
warnings.filterwarnings("ignore", category=UserWarning)

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask

def collate_pairs_and_text(lst_elements: List, ) -> Dict:
    batch = {"motion_feats_0": collate_tensors([el["pose_0"] for el in lst_elements]),
            "motion_feats_1": collate_tensors([el["pose_1"] for el in lst_elements]),
            "length_0": [x["length_0"] for x in lst_elements], 
            "length_1": [x["length_1"] for x in lst_elements],
            "music_0": torch.stack([torch.tensor(x["music_0"].clone().detach().requires_grad_(True)) for x in lst_elements], dim=0),
            "music_1": torch.stack([torch.tensor(x["music_1"].clone().detach().requires_grad_(True)) for x in lst_elements], dim=0),
            "filename": [x["filename"] for x in lst_elements],
            }
    return batch

def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas