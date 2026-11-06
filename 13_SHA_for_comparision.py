class SingleHeadAttention(nn.Module):
    def __init__(self, embedding_dim: int, attention_dim: int):
        super().__init__()
        torch.manual_seed(0)
        self.C   = embedding_dim
        self.d_k = attention_dim
        self.W_K = nn.Linear(self.C, self.d_k, bias=False)
        self.W_Q = nn.Linear(self.C, self.d_k, bias=False)
        self.W_V = nn.Linear(self.C, self.d_k, bias=False)

    def forward(self, embedded):
        B, T, C = embedded.shape
        d_k = self.d_k

        # 1. Project
        K = self.W_K(embedded)      # [B, T, d_k]
        Q = self.W_Q(embedded)      # [B, T, d_k]
        V = self.W_V(embedded)      # [B, T, d_k]

        # 1b. Split heads            -- (none)

        # 2. Scores
        attention_matrix = Q @ K.transpose(-2, -1)   # [B, T, T]
        attention_scores = attention_matrix / (d_k ** 0.5)

        # 3. Causal mask
        mask = torch.tril(torch.ones(T, T)) # [T, T]
        mask = (mask == 0)
        masked_scores = attention_scores.masked_fill(mask, float('-inf'))

        # 4. Softmax
        A = F.softmax(masked_scores, dim=-1)   # [B, T, T]

        # 5. Blend values
        Z = A @ V                   # [B, T, d_k]

        # 6. Merge heads             -- (none)
        # 7. Output projection       -- (none)
        return torch.round(Z, decimals=4)