import torch
import numpy as np

def lengths_to_mask(lengths, max_len):
    # max_len = max(lengths)
    mask = torch.arange(max_len, device=lengths.device).expand(len(lengths), max_len) < lengths.unsqueeze(1)
    return mask
    

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
        
    # if 'text' in notnone_batches[0]:
    #     textbatch = [b['text'] for b in notnone_batches]
    #     cond['y'].update({'text': textbatch})

    # if 'tokens' in notnone_batches[0]:
    #     textbatch = [b['tokens'] for b in notnone_batches]
    #     cond['y'].update({'tokens': textbatch})

    # if 'action' in notnone_batches[0]:
    #     actionbatch = [b['action'] for b in notnone_batches]
    #     cond['y'].update({'action': torch.as_tensor(actionbatch).unsqueeze(1)})

    # # collate action textual names
    # if 'action_text' in notnone_batches[0]:
    #     action_text = [b['action_text']for b in notnone_batches]
    #     cond['y'].update({'action_text': action_text})

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


