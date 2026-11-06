class MultiHeadedSelfAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int, num_heads: int):
        super().__init__()
        torch.manual_seed(0)
        assert attention_dim % num_heads == 0
        self.C    = embedding_dim
        self.h    = num_heads
        self.hd_k = attention_dim
        self.d_k  = attention_dim // num_heads # head dim
        self.W_K = nn.Linear(self.C, self.hd_k, bias=False)
        self.W_Q = nn.Linear(self.C, self.hd_k, bias=False)
        self.W_V = nn.Linear(self.C, self.hd_k, bias=False)
        self.W_O = nn.Linear(self.hd_k, self.C, bias=False)         # optional, see below

    def forward(self, embedded):
        B, T, C = embedded.shape
        h, d_k, hd_k = self.h, self.d_k, self.hd_k

        # 1. Project
        K = self.W_K(embedded)      # [B, T, hd_k]
        Q = self.W_Q(embedded)      # [B, T, hd_k]
        V = self.W_V(embedded)      # [B, T, hd_k]

        # 1b. Split heads
        K = K.view(B, T, h, d_k).transpose(1, 2)  # [B, h, T, d_k]
        Q = Q.view(B, T, h, d_k).transpose(1, 2)  # [B, h, T, d_k]
        V = V.view(B, T, h, d_k).transpose(1, 2)  # [B, h, T, d_k]

        # 2. Scores
        attention_matrix = Q @ K.transpose(-2, -1) # [B, h, T, T]
        attention_scores = attention_matrix / (d_k ** 0.5)

        # 3. Causal mask
        mask = torch.tril(torch.ones(T, T)) # [T, T]
        mask = (mask == 0)
        masked_scores = attention_scores.masked_fill(mask, float('-inf'))

        # 4. Softmax
        A = F.softmax(masked_scores, dim=-1)    # [B, h, T, T]

        # 5. Blend values
        Z = A @ V          # [B, h, T, d_k]

        # 6. Merge heads
        Z = Z.transpose(1, 2).contiguous().view(B, T, hd_k)         # [B, T, hd_k]

        # 7. Output projection
        out = self.W_O(Z)  # [B, T, C]
        return torch.round(out, decimals=4)