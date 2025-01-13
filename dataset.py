import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BiLingualDataset(Dataset):

    def __init__(self, ds, tokenzier_src, tokenzier_tgt, src_lan, tgt_lang, seq_len)->None:
        super().__init__()
        
