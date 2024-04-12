import torch
import numpy as np
from typing import List, Dict

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

def collate_contrastive(lst_elements: List, ) -> Dict:
    concat_lst = [{
        "features": torch.cat((el["pose_0"], el["pose_1_with_transition"]), axis=0),
        "length":   el["length_0"] + el["length_1_with_transition"],
        "music": torch.cat((el["music_0"], el["music_1"]), axis=0),
    } for el in lst_elements]

    batch = {"motion_feats": collate_tensors([el["features"] for el in concat_lst]),
            "length": [x["length"] for x in concat_lst], 
            "music": torch.stack([torch.tensor(x["music"]) for x in lst_elements], dim=0)
            }

    return batch

def collate_pairs_and_text(lst_elements: List, ) -> Dict:
    batch = {"motion_feats_0": collate_tensors([el["pose_0"] for el in lst_elements]),
            "motion_feats_1": collate_tensors([el["pose_1"] for el in lst_elements]),
            "motion_feats_1_with_transition": collate_tensors([el["pose_1_with_transition"] for el in lst_elements]),
            "length_0": [x["length_0"] for x in lst_elements], 
            "length_1": [x["length_1"] for x in lst_elements], 
            "length_transition": [x["length_transition"] for x in lst_elements], 
            "length_1_with_transition": [x["length_1_with_transition"] for x in lst_elements],
            "music_0": torch.stack([torch.tensor(x["music_0"].clone().detach().requires_grad_(True)) for x in lst_elements], dim=0),
            "music_0_with_transition": torch.stack([torch.tensor(x["music_0_with_transition"].clone().detach().requires_grad_(True)) for x in lst_elements], dim=0),
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


def collate(batch):
    # print(batch[0]["pose"])
    # exit()
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b['pose'] for b in notnone_batches]
    databatch1 = [b['pose_1'] for b in notnone_batches]
    # if 'lengths' in notnone_batches[0]:
    #     lenbatch = [b['lengths'] for b in notnone_batches]
    # else:
    #     lenbatch = [len(b['pose'][0]) for b in notnone_batches]
    
    # musiclen = len(notnone_batches[0]['pose'])
    # print(musiclen)
    lenbatch = [len(b['music']) // 2 for b in notnone_batches]
    lenmotion = [len(b['pose'][1]) for b in notnone_batches]
    
    # print(lenbatch)
    
    databatchTensor = collate_tensors(databatch)
    databatchTensor1 = collate_tensors(databatch1)
    lenbatchTensor = torch.as_tensor(lenbatch)
    lenmotionTensor = torch.as_tensor(lenmotion)
    maskbatchTensor = lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1]).unsqueeze(1).unsqueeze(1) # unqueeze for broadcasting

    motion = {'motion_feats_0': databatchTensor, 'motion_feats_1': databatchTensor1}
    cond = {'y': {'mask': maskbatchTensor, 'lengths': lenmotionTensor}}
    if "music" in notnone_batches[0]:
        musicbatch = [b['music'].tolist()[:lenbatch[0]] for b in notnone_batches]
        musicbatch_1 = [b['music'].tolist()[lenbatch[0]:] for b in notnone_batches]
        cond['y'].update({
            'music': torch.tensor(np.array(musicbatch), dtype=torch.float32),
            'music_1': torch.tensor(np.array(musicbatch_1), dtype=torch.float32),
            })

    return motion, cond

# an adapter to our collate func
def t2m_collate(batch):
    # batch.sort(key=lambda x: x[3], reverse=True)
    adapted_batch = [{
        'inp': torch.tensor(b[4].T).float().unsqueeze(1), # [seqlen, J] -> [J, 1, seqlen]
        'text': b[2], #b[0]['caption']
        'tokens': b[6],
        'lengths': b[5],
    } for b in batch]
    return collate(adapted_batch)


