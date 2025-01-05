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

        #create a vector of shape (seq_len,1)
        position = torch.arange(0, seq_len, dtype=torch.float).reshape(seq_len,1)
        #using log space for nujjmerical stability
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model)) 

        #apply sine and cosine for even and odd positions respectively
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        #add batch dimension so that it can applied to whole sentences
        #Adds a new dimension at the 0th index (the beginning of the tensor). 
        #If pe has shape (seq_len, d_model), after applying .unsqueeze(0), the new shape will be (1, seq_len, d_model). This represents a batch of size 1.
        pe = pe.unsqueeze(0)

        #This method registers a tensor as a buffer in the module, which means it will be treated as part of the module but not as a model parameter
        # (i.e., it will not be updated during backpropagation)
        self.register_buffer('pe', pe)

    #apply positional encoding operation to all the words in the sentence
    def forward(self, x):
        x = x + (self.pe[:, x.shape[1], :]).requires_grad_(False)#not learning pe during back prop
        return self.dropout(x)
    
class LayerNormalization(nn.Module):
    def __init__(self, eps : float = 10**-6)->None:
        super().__init__()
        self.eps = eps
        #alpha and beta are parameters
        self.alpha = nn.Parameter(torch.ones(1))
        self.bias = nn.Parameter(torch.zeros(0))

    def forward(self, x):
        #calculating mean and std wrt last dimension ie d_model
        mean = x.mean(dim=-1, keepdim = True)
        std = x.std(dim=-1, keepdim = True)

        return self.alpha * (x - mean)/(std+self.eps) + self.bias
    


#this is a fully connected layer
class FeedForwadBlock(nn.Module):

    def __init__(self, d_model:int, d_ff:int, dropout: float )->None:
        super().__init__()
        self.linear1 = nn.Linear(d_model,d_ff) #W1 and B1->B1 is already defined by tf
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ff, d_model) #w2 and B2

    def forward(self, x):
        #input is tensor with batch,seq_len,d_model --> Linear1 -> batch,seq_len,d_ff -->Linear2 -> batch,seq_len,d_model
        return self.linear2(self.dropout(torch.relu((self.linear1(x)))))

