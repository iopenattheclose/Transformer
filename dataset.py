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

    def __len__(self):
        return len(self.ds)
    
    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]
        src_text = src_tgt_pair['translation'][self.src_lan]
        tgt_text = src_tgt_pair['translation'][self.tgt_lang]

        enc_input_tokens = self.tokenzier_src.encode(src_text).ids
        dec_input_tokens  = self.tokenzier_tgt.encode(tgt_text).ids

        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 1

        if(enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0):
            raise ValueError('Sentence is too long')

        #add sos and eos to src text tokens
        encoder_input = torch.cat(
            self.sos_token,
            torch.tensor(enc_num_padding_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token] * enc_num_padding_tokens,dtype=torch.int64)
        )

        decoder_input = torch.concat(
            self.sos_token,
            torch.tensor(dec_num_padding_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token] * dec_num_padding_tokens,dtype=torch.int64)
        )

        label = torch.cat(
            
        )


