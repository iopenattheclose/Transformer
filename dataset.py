import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BiLingualDataset(Dataset):

    def __init__(self, ds, tokenzier_src, tokenzier_tgt, src_lan, tgt_lang, seq_len)->None:
        super().__init__()

        self.ds = ds
        self.tokenzier_src = tokenzier_src
        self.tokenzier_tgt = tokenzier_tgt
        self.src_lan = src_lan
        self.tgt_lang = tgt_lang

        self.sos_token = torch.Tensor([tokenzier_src.token_to_id(['[SOS]'])], dtype = torch.int64)
        self.eos_token = torch.Tensor([tokenzier_src.token_to_id(['[EOS]'])], dtype = torch.int64)
        self.pad_token = torch.Tensor([tokenzier_src.token_to_id(['[PAD]'])], dtype = torch.int64)

        



