import torch
import torch.nn as nn
from torchtyping import TensorType

class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        self.C = embedding_dim
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.d_k = attention_dim // num_heads
        self.projection = nn.Linear(attention_dim, attention_dim, bias = False)

        self.W_K = nn.Linear(self.C, self.d_k, bias = False)
        self.W_Q = nn.Linear(self.C, self.d_k, bias = False)
        self.W_V = nn.Linear(self.C, self.d_k, bias = False)

    def forward(self, embedded:TensorType[float]) -> TensorType[float]:
        # Temporary dictionary ( His to Mine conversion)
        # D -> C        : Embedding dimension 
        # H -> h        : num_heads aka number of heads
        # C -> d_k      : head dimension. This is the dimension of K, Q & V vectors of each word.

        # embedded is [B, T, C]
        # Output of each head is [B, T, d_k]
        # Output of MHA is [B,T, A] before W_O, [B,T, C] after W_O, where A = hd_k = attention_dim

        B, T, C = embedded.shape
        h = self.num_heads              # Number of attention heads = h
        d_k = self.d_k                 # Head dimension - d_k
        hd_k = self.attention_dim       # Dimension of the output vector of each token, 
        # where all the attention heads are concatted.
        # There are "h" heads and dimention of each is d_k, so h*d_k is the total attention_dim.
        # attention_dim is the dimension of the output in the Attention Layer before W_O. 
        # This is the concatenated matrix of the output of each of the Single SelfAttention heads.

        # 1.Linear Projections and reshape
        # [B, T, C ] -> [B, T, h, d_k] -> [B, h, T, d_k] - why tho?
        K = self.W_K(embedded) # Shape = [B, T, d_k]
        Q = self.W_Q(embedded) # Shape = [B, T, d_k]
        V = self.W_V(embedded) # Shape = [B, T, d_k]



