"""
Multi-head self-attention by hand: no nn.Module, no nn.Linear.
Same sentence, same X, same weights as the Step 0-9 walkthrough.

Weights are written in x out (paper style), so every projection is X @ W.
At each step it prints shape, stride and is_contiguous() so you can watch
view / transpose / contiguous change the strides.

Predict each stride tuple BEFORE you run it.
"""
import torch
import torch.nn.functional as F

torch.set_printoptions(precision=2, sci_mode=False)


def show(name, t):
    print(f"\n=== {name}   shape={tuple(t.shape)}   stride={t.stride()}   contiguous={t.is_contiguous()}")
    print(t)


# ---------------------------------------------------------------- Step 0: setup
# T = 5 tokens, C = 3, one sentence (B = 1)
X = torch.tensor([[1., 0., 1.],   # The
                  [0., 2., 1.],   # tall
                  [1., 1., 0.],   # man
                  [2., 0., 1.],   # runs
                  [0., 1., 2.]])  # .
X = X.unsqueeze(0)                # [B, T, C] = [1, 5, 3]

B, T, C = X.shape
h = 2                 # number of heads
d_k = 2               # size of one head
hd_k = h * d_k        # attention_dim = 4

# ---------------------------------------------------------------- Step 1: weights
# C x hd_k = 3 x 4. Columns 0-1 = head 0, columns 2-3 = head 1.
#                    head 0  | head 1
W_Q = torch.tensor([[1., 0.,   1.,  0.],
                    [0., 1.,   0.,  0.],
                    [1., 1.,   0., -1.]])
W_K = torch.tensor([[0., 1.,   1.,  0.],
                    [1., 0.,   1.,  0.],
                    [1., 0.,  -1.,  1.]])
W_V = torch.tensor([[1., 0.,   2.,  0.],
                    [0., 1.,   0.,  1.],
                    [1., 0.,   0.,  2.]])
# hd_k x C = 4 x 3. Rows 0-1 = head 0's W^O_0, rows 2-3 = head 1's W^O_1.
W_O = torch.tensor([[1., 0., 1.],
                    [0., 1., 0.],
                    [1., 0., 0.],
                    [0., 1., 1.]])

# ---------------------------------------------------------------- Step 2: project (all heads in one matmul)
Q = X @ W_Q                       # [B, T, hd_k]
K = X @ W_K
V = X @ W_V
show("Step 2: Q_f = X @ W_Q", Q)
show("Step 2: K_f = X @ W_K", K)
show("Step 2: V_f = X @ W_V", V)

# ---------------------------------------------------------------- Step 3: split heads
Q = Q.view(B, T, h, d_k)          # [B, T, h, d_k]   no copy
show("Step 3a: Q.view(B, T, h, d_k)", Q)
Q = Q.transpose(1, 2)             # [B, h, T, d_k]   no copy, strides swap
show("Step 3b: Q.transpose(1, 2)", Q)

K = K.view(B, T, h, d_k).transpose(1, 2)
V = V.view(B, T, h, d_k).transpose(1, 2)
show("Step 3: K split", K)
show("Step 3: V split", V)

# What goes wrong if you skip the transpose and view straight to [B, h, T, d_k]:
Q_wrong = (X @ W_Q).view(B, h, T, d_k)
show("WRONG: (X @ W_Q).view(B, h, T, d_k)  -- 'head 0' mixes tokens and heads", Q_wrong)

# ---------------------------------------------------------------- Step 4: scores
attention_matrix = Q @ K.transpose(-2, -1)          # [B, h, T, T]
attention_scores = attention_matrix / (d_k ** 0.5)
show("Step 4: attention_scores", attention_scores)

# ---------------------------------------------------------------- Step 5: causal mask
mask = torch.tril(torch.ones(T, T))                 # [T, T], broadcasts over [B, h]
mask = (mask == 0)
masked_scores = attention_scores.masked_fill(mask, float('-inf'))
show("Step 5: masked_scores", masked_scores)

# ---------------------------------------------------------------- Step 6: softmax, one A per head
A = F.softmax(masked_scores, dim=-1)                # [B, h, T, T]
show("Step 6: A", A)
print("row sums:", A.sum(dim=-1))

# ---------------------------------------------------------------- Step 7: blend values
Z = A @ V                                           # [B, h, T, d_k]
show("Step 7: Z = A @ V", Z)

# ---------------------------------------------------------------- Step 8: merge heads
Z_t = Z.transpose(1, 2)                             # [B, T, h, d_k]   no copy
show("Step 8a: Z.transpose(1, 2)", Z_t)

print("\nman's 4 numbers in memory (flat offsets):")
flat = Z.flatten()                                  # Z itself is contiguous, so this is its memory
for off in (4, 5, 14, 15):
    print(f"  offset {off:2d} -> {flat[off].item():.2f}")

try:
    Z_t.view(B, T, hd_k)
except RuntimeError as e:
    print("\nStep 8b: Z_t.view(B, T, hd_k) FAILED:\n ", e)

Z_c = Z_t.contiguous()                              # copy into token-major order
show("Step 8c: Z_t.contiguous()", Z_c)
Z_merged = Z_c.view(B, T, hd_k)                     # [B, T, hd_k]   free now
show("Step 8d: .view(B, T, hd_k)", Z_merged)

# ---------------------------------------------------------------- Step 9: output projection
out = Z_merged @ W_O                                # [B, T, C]
show("Step 9: out = Z_merged @ W_O", out)

# ---------------------------------------------------------------- checks
# out is the sum of each head's own projection
head_parts = Z[:, 0] @ W_O[:2] + Z[:, 1] @ W_O[2:]
print("\nout == Z_0 @ W^O_0 + Z_1 @ W^O_1 ?", torch.allclose(out, head_parts))

# head 0 reproduces Part 2's single-head attention (same W_Q, W_K; Part 2's 3x4 W_V == our fused W_V)
Q2, K2, V2 = X[0] @ W_Q[:, :2], X[0] @ W_K[:, :2], X[0] @ W_V
S2 = (Q2 @ K2.T / d_k ** 0.5).masked_fill(mask, float('-inf'))
Z_part2 = F.softmax(S2, dim=-1) @ V2
print("head 0's A == Part 2's A ?", torch.allclose(A[0, 0], F.softmax(S2, dim=-1)))
print("Z_merged[:, :2] == Part 2's Z[:, :2] ?", torch.allclose(Z_merged[0, :, :2], Z_part2[:, :2]))

# ---------------------------------------------------------------- bonus: your h = 12, d_k = 64 case
print("\n\n=== Bonus: h = 12, d_k = 64, T = 5 (strides only)")
Z12 = torch.randn(1, 12, 5, 64)                     # A @ V output, [B, h, T, d_k]
Z12_t = Z12.transpose(1, 2)                         # [B, T, h, d_k]
print("A @ V          ", tuple(Z12.shape), Z12.stride())
print("transpose(1, 2)", tuple(Z12_t.shape), Z12_t.stride(), "contiguous =", Z12_t.is_contiguous())
try:
    Z12_t.view(1, 5, 768)
except RuntimeError as e:
    print("view(1, 5, 768) FAILED:", e)
Z12_c = Z12_t.contiguous()
print("contiguous()   ", tuple(Z12_c.shape), Z12_c.stride())
print("view(1, 5, 768)", tuple(Z12_c.view(1, 5, 768).shape), Z12_c.view(1, 5, 768).stride())
