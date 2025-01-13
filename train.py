import torch
import torch.nn as nn

import datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path

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