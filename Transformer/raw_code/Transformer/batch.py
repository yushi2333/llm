from torchtext.legacy import data 
import torch
from torch.autograd import Variable # Deprecated, but keeping for compatibility
import numpy as np

def nopeak_mask(size, device):
    np_mask = np.triu(np.ones((1, size, size)), k=1).astype('uint8')
    np_mask = (torch.from_numpy(np_mask == 0)).to(device) # Move mask to device
    # Using Variable is deprecated, but keeping for compatibility
    # np_mask = Variable(np_mask)
    return np_mask

def create_masks(src, trg, src_pad, trg_pad):
    
    src_mask = (src != src_pad).unsqueeze(-2)

    if trg is not None:
        trg_mask = (trg != trg_pad).unsqueeze(-2)
        size = trg.size(1)
        # Pass the device of trg to nopeak_mask
        np_mask = nopeak_mask(size, trg.device)
        trg_mask = trg_mask & np_mask

    else:
        trg_mask = None
    return src_mask, trg_mask 