import torch
import torch.nn as nn
import math

class InputEmbeddings(nn.Module):

    def __init__(self, d_model: int, vocab_size: int):
        super().__init__()
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(vocab_size,d_model)

    def forward(self, x):
        # Apply the embedding layer
        embeddings = self.embedding(x) * math.sqrt(self.d_model)

    #positional encoding
    #adding another vector of size 512 to each word using the PE formula

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, seq_len: int, dropout :float) -> None:
        super().__init__()
        self.d_model = d_model
        self.seq_len = seq_len
        self.dropout = nn.Dropout(dropout)

        #Create a matrix of shape d_model, seq_len (this will be added to the word embeddings)
        pe = torch.zeros(seq_len, d_model)
