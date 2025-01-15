import torch
import torch.nn as nn

import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

from model import build_transformer

import warnings
from tqdm import tqdm
import os
from pathlib import Path

from dataset import BiLingualDataset,causal_mask

# Huggingface datasets and tokenizers
from datasets import load_dataset
from tokenizer import Tokenizer
from tokenizer import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def get_or_build_tokenizer(config, dataset, language):
    tokenizer_path = Path(config['tokenizer_file'].format(language))
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token='[UNK]'))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]","[PAD]","[SOS]","[EOS]"], min_frequency = 2)
        tokenizer.train_from_iterator(get_all_sentences(dataset,language),trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    return tokenizer

def get_all_sentences(dataset, language):
    for item in dataset:
        yield item['translation'][language]


def get_dataset(config):
    ds_raw = load_dataset('opus_books',f'{config["lang_src"]}-{config["lang_tgt"]}', split='train')

    #bbuild tokenizer
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    #train test split 80 20 or 90 10
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw,[train_ds_size, val_ds_size])

    train_ds = BiLingualDataset(train_ds_raw, tokenizer_src, config['lang_src'],config['tgt_src'],config['seq_len'])
    val_ds = BiLingualDataset(val_ds_raw, tokenizer_src, config['lang_src'],config['tgt_src'],config['seq_len'])

    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_src.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt= max(max_len_tgt, len(tgt_ids))

        print(f'Max length of source and target sentence is {0} and {1} respectively',max_len_src, max_len_tgt)

    train_dataloader = DataLoader(train_ds, batch_size = config['batch_size'], shuffle=True)
    val_dataloader = DataLoader(val_ds, batch_size = 1, shuffle=True)

    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], config['d_model'])
    return model






    




# def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
#     sos_idx = tokenizer_tgt.token_to_id('[SOS]')
#     eos_idx = tokenizer_tgt.token_to_id('[EOS]')

#     # Precompute the encoder output and reuse it for every step
#     encoder_output = model.encode(source, source_mask)
#     # Initialize the decoder input with the sos token
#     decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
#     while True:
#         if decoder_input.size(1) == max_len:
#             break

#         # build mask for target
#         decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

#         # calculate output
#         out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

#         # get next token
#         prob = model.project(out[:, -1])
#         _, next_word = torch.max(prob, dim=1)
#         decoder_input = torch.cat(
#             [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
#         )

#         if next_word == eos_idx:
#             break

#     return decoder_input.squeeze(0)

#     def run_validation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, print_msg, global_step, writer, num_examples=2):
#         model.eval()
#         count = 0

#         source_texts = []
#         expected = []
#         predicted = []

#         try:
#             pass
#         except:
#             pass