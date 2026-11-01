import os
import torch
# Print the NVIDIA CUDA visible devices environment variable
print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))

# Check if CUDA is available
cuda_available = torch.cuda.is_available()
print("Is CUDA available?:", cuda_available)

if cuda_available:
    print("GPU Device Count:", torch.cuda.device_count())
    print("Primary GPU Name:", torch.cuda.get_device_name(0))
else:
    print("No NVIDIA GPU detected by PyTorch.")

import torch
import torch.nn.functional as F

# T = 5 tokens, d_model = 3
X = torch.tensor([[1., 0., 1.],   # The
                  [0., 2., 1.],   # tall
                  [1., 1., 0.],   # man
                  [2., 0., 1.],   # runs
                  [0., 1., 2.]])  # .

print(X)

# Weights as in × out, like the paper (d_k = 2, d_v = 4)
W_Q = torch.tensor([[1., 0.], [0., 1.], [1., 1.]])                          # 3 × 2
print(f"W_Q:")
print(W_Q)
W_K = torch.tensor([[0., 1.], [1., 0.], [1., 0.]])                          # 3 × 2
print(f"W_K:")
print(W_K)
W_V = torch.tensor([[1., 0., 2., 0.], [0., 1., 0., 1.], [1., 0., 0., 2.]])  # 3 × 4
print(f"W_V:")
print(W_V)

Q = X @ W_Q       # 5 x 2 - query and key vectors are 2 dimensional.
K = X @ W_K       # 5 x 2 - I did this so that all numbers like T = 5, d_k = 2, d_v = 4 d_model = 3 are all different and easy to see.
V = X @ W_V       # 5 x 4 - value vector is 4 dimensional, v1, v2, v3, v4

T, d_k = Q.shape                # T =5, d_k = 2

Attention_matrix = Q @ K.T
print(Attention_matrix.shape)
print(Attention_matrix)

# Scale attention matrix by len of query vector. I don't know why this is done
# But I will do it now regardless.
# BTW, len of query vector is same as len of key vector. => d_q = d_k
# In this case the query and the key vectors emitted are of length 2
# So d_k = 2

Attention_scores = Attention_matrix / (d_k ** 0.5)

print(Attention_scores.shape)

#Boilerplate to print numbers with 1 decimal places temporarily
old_precision = torch._tensor_str.PRINT_OPTS.precision # Save old precision
torch.set_printoptions(precision=1)                    # Temporarily modify it

print(Attention_scores)

torch.set_printoptions(precision=old_precision)       # Change it back

allowed = torch.tril(torch.ones(T, T, dtype=torch.bool)) # T was 5
print(allowed)

"""torch.tril(...) keeps the lower triangle, including the diagonal, and zeros the rest. So allowed[t, s] is True when $s \le t$: token $t$ may look at itself and everything before it."""

Attention_scores = Attention_scores.masked_fill(~allowed, float('-inf'))
# Seems like this does a msked fill on attention_scores based on the mask called "allowed". In allowed the top half was blacked out.
# So here we take the opposite of that "~allowed". That inverts the mask. Now we have selected the top right part of the mask as True. Then we put -inf on that newly selected part which is the top-right.
# This ultimately means that we have created a mask of -inf for the top right triangle.
# This is called the causal mask.
# 2. causal mask: True on and below the diagonal = "allowed to look"
print(Attention_scores)

# Now we can take the softmax
A = F.softmax(Attention_scores, dim = 1)
print(A)

# 4. Take A.V

Z = A @ V       # 5x4
print(Z)        # Z is the output of the attention layer,
# Actually Z is the output of one slef-Attention head. Several outputs from many such heads are combined and formed into the actual output of the attention layer. This is followed up by a MLP in the Transformer Block.

