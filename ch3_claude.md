# 3 Transformer Language Model Architecture

A language model takes as input a batched sequence of integer token IDs (i.e., `torch.Tensor` of shape `(batch_size, sequence_length)`), and returns a (batched) normalized probability distribution over the vocabulary (i.e., a PyTorch Tensor of shape `(batch_size, sequence_length, vocab_size)`), where the predicted distribution is over the next word for each input token. When training the language model, we use these next-word predictions to calculate the cross-entropy loss between the actual next word and the predicted next word. When generating text from the language model during inference, we take the predicted next-word distribution from the final time step (i.e., the last item in the sequence) to generate the next token in the sequence (e.g., by taking the token with the highest probability, sampling from the distribution, etc.), add the generated token to the input sequence, and repeat.

In this part of the assignment, you will build this Transformer language model from scratch. We will begin with a high-level description of the model before progressively detailing the individual components.

## 3.1 Transformer LM

Given a sequence of token IDs, the Transformer language model uses an input embedding to convert token IDs to dense vectors, passes the embedded tokens through `num_layers` Transformer blocks, and then applies a learned linear projection (the "output embedding" or "LM head") to produce the predicted next-token logits. See Figure 1 for a schematic representation.

![Transformer LM Overview](figure1-placeholder)
*Figure 1: An overview of our Transformer language model.*

### 3.1.1 Token Embeddings

In the very first step, the Transformer embeds the (batched) sequence of token IDs into a sequence of vectors containing information on the token identity (red blocks in Figure 1).

More specifically, given a sequence of token IDs, the Transformer language model uses a token embedding layer to produce a sequence of vectors. Each embedding layer takes in a tensor of integers of shape `(batch_size, sequence_length)` and produces a sequence of vectors of shape `(batch_size, sequence_length, d_model)`.

### 3.1.2 Pre-norm Transformer Block

After embedding, the activations are processed by several identically structured neural net layers. A standard decoder-only Transformer language model consists of `num_layers` identical layers (commonly called Transformer "blocks"). Each Transformer block takes in an input of shape `(batch_size, sequence_length, d_model)` and returns an output of shape `(batch_size, sequence_length, d_model)`. Each block aggregates information across the sequence (via self-attention) and non-linearly transforms it (via the feed-forward layers).

## 3.2 Output Normalization and Embedding

After `num_layers` Transformer blocks, we will take the final activations and turn them into a distribution over the vocabulary.

We will implement the "pre-norm" Transformer block (detailed in §3.5), which additionally requires the use of layer normalization (detailed below) after the final Transformer block to ensure its outputs are properly scaled.

After this normalization, we will use a standard learned linear transformation to convert the output of the Transformer blocks into predicted next-token logits (see, e.g., Radford et al. [2018] equation 2).

## 3.3 Remark: Batching, Einsum and Efficient Computation

Throughout the Transformer, we will be performing the same computation applied to many batch-like inputs. Here are a few examples:

- **Elements of a batch**: we apply the same Transformer forward operation on each batch element.
- **Sequence length**: the "position-wise" operations like RMSNorm and feed-forward operate identically on each position of a sequence.
- **Attention heads**: the attention operation is batched across attention heads in a "multi-headed" attention operation.

It is useful to have an ergonomic way of performing such operations in a way that fully utilizes the GPU, and is easy to read and understand. Many PyTorch operations can take in excess "batch-like" dimensions at the start of a tensor and repeat/broadcast the operation across these dimensions efficiently.

For instance, say we are doing a position-wise, batched operation. We have a "data tensor" D of shape `(batch_size, sequence_length, d_model)`, and we would like to do a batched vector-matrix multiply against a matrix A of shape `(d_model, d_model)`. In this case, `D @ A` will do a batched matrix multiply, which is an efficient primitive in PyTorch, where the `(batch_size, sequence_length)` dimensions are batched over.

Because of this, it is helpful to assume that your functions may be given additional batch-like dimensions and to keep those dimensions at the start of the PyTorch shape. To organize tensors so they can be batched in this manner, they might need to be shaped using many steps of view, reshape and transpose. This can be a bit of a pain, and it often gets hard to read what the code is doing and what the shapes of your tensors are.

A more ergonomic option is to use einsum notation within `torch.einsum`, or rather use framework agnostic libraries like `einops` or `einx`. The two key ops are `einsum`, which can do tensor contractions with arbitrary dimensions of input tensors, and `rearrange`, which can reorder, concatenate, and split arbitrary dimensions. It turns out almost all operations in machine learning are some combination of dimension juggling and tensor contraction with the occasional (usually pointwise) nonlinear function. This means that a lot of your code can be more readable and flexible when using einsum notation.

We strongly recommend learning and using einsum notation for the class. Students who have not been exposed to einsum notation before should use `einops` (docs here), and students who are already comfortable with `einops` should learn the more general `einx` (here). Both packages are already installed in the environment we've supplied.

Here we give some examples of how einsum notation can be used. These are a supplement to the documentation for einops, which you should read first.

### Example: Batched matrix multiplication with einops.einsum

```python
import torch
from einops import rearrange, einsum

## Basic implementation
Y = D @ A.T
# Hard to tell the input and output shapes and what they mean.
# What shapes can D and A have, and do any of these have unexpected behavior?

## Einsum is self-documenting and robust
# D A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## Or, a batched version where D can have any leading dimensions but A is constrained.
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

### Example: Broadcasted operations with einops.rearrange

We have a batch of images, and for each image we want to generate 10 dimmed versions based on some scaling factor:

```python
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## Reshape and multiply
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## Or in one go:
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

### Example: Pixel mixing with einops.rearrange

Suppose we have a batch of images represented as a tensor of shape `(batch, height, width, channel)`, and we want to perform a linear transformation across all pixels of the image, but this transformation should happen independently for each channel. Our linear transformation is represented as a matrix B of shape `(height × width, height × width)`.

```python
channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## Rearrange an image tensor for mixing across all pixels
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

Instead, using einops:

```python
height = width = 32
## Rearrange replaces clunky torch view + transpose
channels_first = rearrange(
    channels_last,
    "batch height width channel -> batch channel (height width)"
)
channels_first_transformed = einsum(
    channels_first, B,
    "batch channel pixel_in, pixel_out pixel_in -> batch channel pixel_out"
)
channels_last_transformed = rearrange(
    channels_first_transformed,
    "batch channel (height width) -> batch height width channel",
    height=height, width=width
)
```

Or, if you're feeling crazy: all in one go using `einx.dot` (einx equivalent of `einops.einsum`):

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

The first implementation here could be improved by placing comments before and after to indicate what the input and output shapes are, but this is clunky and susceptible to bugs. With einsum notation, documentation is implementation!

Einsum notation can handle arbitrary input batching dimensions, but also has the key benefit of being self-documenting. It's much clearer what the relevant shapes of your input and output tensors are in code that uses einsum notation. For the remaining tensors, you can consider using Tensor type hints, for instance using the `jaxtyping` library (not specific to Jax).

We will talk more about the performance implications of using einsum notation in assignment 2, but for now know that they're almost always better than the alternative!

### 3.3.1 Mathematical Notation and Memory Ordering

Many machine learning papers use row vectors in their notation, which result in representations that mesh well with the row-major memory ordering used by default in NumPy and PyTorch. With row vectors, a linear transformation looks like

```
y = xW⊤                                                                  (1)
```

for row-major W ∈ R^(d_out×d_in) and row-vector x ∈ R^(1×d_in).

In linear algebra it's generally more common to use column vectors, where linear transformations look like

```
y = Wx                                                                   (2)
```

given a row-major W ∈ R^(d_out×d_in) and column-vector x ∈ R^d_in. We will use column vectors for mathematical notation in this assignment, as it is generally easier to follow the math this way. You should keep in mind that if you want to use plain matrix multiplication notation, you will have to apply matrices using the row vector convention, since PyTorch uses row-major memory ordering. If you use einsum for your matrix operations, this should be a non-issue.

## 3.4 Basic Building Blocks: Linear and Embedding Modules

### 3.4.1 Parameter Initialization

Training neural networks effectively often requires careful initialization of the model parameters—bad initializations can lead to undesirable behavior such as vanishing or exploding gradients. Pre-norm transformers are unusually robust to initializations, but they can still have a significant impact on training speed and convergence. Since this assignment is already long, we will save the details for assignment 3, and instead give you some approximate initializations that should work well for most cases. For now, use:

- **Linear weights**: N(μ = 0, σ² = 2/(d_in+d_out)) truncated at [-3σ, 3σ].
- **Embedding**: N(μ = 0, σ² = 1) truncated at [-3, 3]
- **RMSNorm**: 1

You should use `torch.nn.init.trunc_normal_` to initialize the truncated normal weights.

### 3.4.2 Linear Module

Linear layers are a fundamental building block of Transformers and neural nets in general. First, you will implement your own Linear class that inherits from `torch.nn.Module` and performs a linear transformation:

```
y = Wx                                                                   (3)
```

Note that we do not include a bias term, following most modern LLMs.

**Problem (linear): Implementing the linear module (1 point)**

Deliverable: Implement a Linear class that inherits from `torch.nn.Module` and performs a linear transformation. Your implementation should follow the interface of PyTorch's built-in `nn.Linear` module, except for not having a bias argument or parameter. We recommend the following interface:

```python
def __init__(self, in_features, out_features, device=None, dtype=None):
    """Construct a linear transformation module. This function should accept the following parameters:
    
    in_features: int - final dimension of the input
    out_features: int - final dimension of the output
    device: torch.device | None = None - Device to store the parameters on
    dtype: torch.dtype | None = None - Data type of the parameters
    """

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply the linear transformation to the input."""
```

Make sure to:
- subclass nn.Module
- call the superclass constructor
- construct and store your parameter as W (not W⊤) for memory ordering reasons, putting it in an nn.Parameter
- of course, don't use nn.Linear or nn.functional.linear

For initializations, use the settings from above along with `torch.nn.init.trunc_normal_` to initialize the weights.

To test your Linear module, implement the test adapter at [adapters.run_linear]. The adapter should load the given weights into your Linear module. You can use Module.load_state_dict for this purpose. Then, run `uv run pytest -k test_linear`.

### 3.4.3 Embedding Module

As discussed above, the first layer of the Transformer is an embedding layer that maps integer token IDs into a vector space of dimension d_model. We will implement a custom Embedding class that inherits from `torch.nn.Module` (so you should not use nn.Embedding). The forward method should select the embedding vector for each token ID by indexing into an embedding matrix of shape `(vocab_size, d_model)` using a `torch.LongTensor` of token IDs with shape `(batch_size, sequence_length)`.

**Problem (embedding): Implement the embedding module (1 point)**

Deliverable: Implement the Embedding class that inherits from `torch.nn.Module` and performs an embedding lookup. Your implementation should follow the interface of PyTorch's built-in `nn.Embedding` module. We recommend the following interface:

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    """Construct an embedding module. This function should accept the following parameters:
    
    num_embeddings: int - Size of the vocabulary
    embedding_dim: int - Dimension of the embedding vectors, i.e., d_model
    device: torch.device | None = None - Device to store the parameters on
    dtype: torch.dtype | None = None - Data type of the parameters
    """

def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    """Lookup the embedding vectors for the given token IDs."""
```

Make sure to:
- subclass nn.Module
- call the superclass constructor
- initialize your embedding matrix as a nn.Parameter
- store the embedding matrix with the d_model being the final dimension
- of course, don't use nn.Embedding or nn.functional.embedding

Again, use the settings from above for initialization, and use `torch.nn.init.trunc_normal_` to initialize the weights.

To test your implementation, implement the test adapter at [adapters.run_embedding]. Then, run `uv run pytest -k test_embedding`.

## 3.5 Pre-Norm Transformer Block

Each Transformer block has two sub-layers: a multi-head self-attention mechanism and a position-wise feed-forward network (Vaswani et al., 2017, section 3.1).

In the original Transformer paper, the model uses a residual connection around each of the two sub-layers, followed by layer normalization. This architecture is commonly known as the "post-norm" Transformer, since layer normalization is applied to the sublayer output. However, a variety of work has found that moving layer normalization from the output of each sub-layer to the input of each sub-layer (with an additional layer normalization after the final Transformer block) improves Transformer training stability [Nguyen and Salazar, 2019, Xiong et al., 2020]—see Figure 2 for a visual representation of this "pre-norm" Transformer block. The output of each Transformer block sub-layer is then added to the sub-layer input via the residual connection (Vaswani et al., 2017, section 5.4). An intuition for pre-norm is that there is a clean "residual stream" without any normalization going from the input embeddings to the final output of the Transformer, which is purported to improve gradient flow. This pre-norm Transformer is now the standard used in language models today (e.g., GPT-3, LLaMA, PaLM, etc.), so we will implement this variant. We will walk through each of the components of a pre-norm Transformer block, implementing them in sequence.

![Pre-norm Transformer Block](figure2-placeholder)
*Figure 2: A pre-norm Transformer block.*

### 3.5.1 Root Mean Square Layer Normalization

The original Transformer implementation of Vaswani et al. [2017] uses layer normalization [Ba et al., 2016] to normalize activations. Following Touvron et al. [2023], we will use root mean square layer normalization (RMSNorm; Zhang and Sennrich, 2019, equation 4) for layer normalization. Given a vector a ∈ R^d_model of activations, RMSNorm will rescale each activation a_i as follows:

```
RMSNorm(a_i) = (a_i / RMS(a)) * g_i                                     (4)
```

where `RMS(a) = √(1/d_model * Σ(a_i²) + ε)`. Here, g_i is a learnable "gain" parameter (there are d_model such parameters total), and ε is a hyperparameter that is often fixed at 1e-5.

You should upcast your input to `torch.float32` to prevent overflow when you square the input. Overall, your forward method should look like:

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# Your code here performing RMSNorm
...
result = ...
# Return the result in the original dtype
return result.to(in_dtype)
```

**Problem (rmsnorm): Root Mean Square Layer Normalization (1 point)**

Deliverable: Implement RMSNorm as a `torch.nn.Module`. We recommend the following interface:

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    """Construct the RMSNorm module. This function should accept the following parameters:
    
    d_model: int - Hidden dimension of the model
    eps: float = 1e-5 - Epsilon value for numerical stability
    device: torch.device | None = None - Device to store the parameters on
    dtype: torch.dtype | None = None - Data type of the parameters
    """

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Process an input tensor of shape (batch_size, sequence_length, d_model) 
    and return a tensor of the same shape."""
```

Note: Remember to upcast your input to `torch.float32` before performing the normalization (and later downcast to the original dtype), as described above.

To test your implementation, implement the test adapter at [adapters.run_rmsnorm]. Then, run `uv run pytest -k test_rmsnorm`.

### 3.5.2 Position-Wise Feed-Forward Network

![SiLU vs ReLU](figure3-placeholder)
*Figure 3: Comparing the SiLU (aka Swish) and ReLU activation functions.*

In the original Transformer paper (section 3.3 of Vaswani et al. [2017]), the Transformer feed-forward network consists of two linear transformations with a ReLU activation (ReLU(x) = max(0, x)) between them. The dimensionality of the inner feed-forward layer is typically 4x the input dimensionality.

However, modern language models tend to incorporate two main changes compared to this original design: they use another activation function and employ a gating mechanism. Specifically, we will implement the "SwiGLU" activation function adopted in LLMs like Llama 3 [Grattafiori et al., 2024] and Qwen 2.5 [Yang et al., 2024], which combines the SiLU (often called Swish) activation with a gating mechanism called a Gated Linear Unit (GLU). We will also omit the bias terms sometimes used in linear layers, following most modern LLMs since PaLM [Chowdhery et al., 2022] and LLaMA [Touvron et al., 2023].

The SiLU or Swish activation function [Hendrycks and Gimpel, 2016, Elfwing et al., 2017] is defined as follows:

```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))                                   (5)
```

As can be seen in Figure 3, the SiLU activation function is similar to the ReLU activation function, but is smooth at zero.

Gated Linear Units (GLUs) were originally defined by Dauphin et al. [2017] as the element-wise product of a linear transformation passed through a sigmoid function and another linear transformation:

```
GLU(x, W1, W2) = σ(W1x) ⊙ W2x                                          (6)
```

where ⊙ represents element-wise multiplication. Gated Linear Units are suggested to "reduce the vanishing gradient problem for deep architectures by providing a linear path for the gradients while retaining non-linear capabilities."

Putting the SiLU/Swish and GLU together, we get the SwiGLU, which we will use for our feed-forward networks:

```
FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)                  (7)
```

where x ∈ R^d_model, W1, W3 ∈ R^(d_ff×d_model), W2 ∈ R^(d_model×d_ff), and canonically, d_ff = (8/3)d_model.

Shazeer [2020] first proposed combining the SiLU/Swish activation with GLUs and conducted experiments showing that SwiGLU outperforms baselines like ReLU and SiLU (without gating) on language modeling tasks. Later in the assignment, you will compare SwiGLU and SiLU. Though we've mentioned some heuristic arguments for these components (and the papers provide more supporting evidence), it's good to keep an empirical perspective: a now famous quote from Shazeer's paper is

> We offer no explanation as to why these architectures seem to work; we attribute their success, as all else, to divine benevolence.

**Problem (positionwise_feedforward): Implement the position-wise feed-forward network (2 points)**

Deliverable: Implement the SwiGLU feed-forward network, composed of a SiLU activation function and a GLU.

Note: in this particular case, you should feel free to use `torch.sigmoid` in your implementation for numerical stability.

You should set d_ff to approximately (8/3) × d_model in your implementation, while ensuring that the dimensionality of the inner feed-forward layer is a multiple of 64 to make good use of your hardware. To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_swiglu]. Then, run `uv run pytest -k test_swiglu` to test your implementation.

### 3.5.3 Relative Positional Embeddings

To inject positional information into the model, we will implement Rotary Position Embeddings [Su et al., 2021], often called RoPE. For a given query token q^(i) = W_q x^(i) ∈ R^d at token position i, we will apply a pairwise rotation matrix R_i, giving us q'^(i) = R_i q^(i) = R_i W_q x^(i). Here, R_i will rotate pairs of embedding elements q^(i)_{2k-1:2k} as 2d vectors by the angle θ_{i,k} = i/Θ^{(2k-1)/d} for k ∈ {1, ..., d/2} and some constant Θ. Thus, we can consider R_i to be a block-diagonal matrix of size d × d, with blocks R^i_k for k ∈ {1, ..., d/2}, with

```
R^i_k = [cos(θ_{i,k})  -sin(θ_{i,k})]                                   (8)
        [sin(θ_{i,k})   cos(θ_{i,k})]
```

Thus we get the full rotation matrix

```
R^i = [R^i_1   0      0    ...  0    ]                                  (9)
      [0       R^i_2   0    ...  0    ]
      [0       0       R^i_3 ...  0    ]
      [...     ...     ... ...   ...  ]
      [0       0       0    ... R^i_{d/2}]
```

where 0s represent 2 × 2 zero matrices. While one could construct the full d × d matrix, a good solution should use the properties of this matrix to implement the transformation more efficiently. Since we only care about the relative rotation of tokens within a given sequence, we can reuse the values we compute for cos(θ_{i,k}) and sin(θ_{i,k}) across layers, and different batches. If you would like to optimize it, you may use a single RoPE module referenced by all layers, and it can have a 2d pre-computed buffer of sin and cos values created during init with `self.register_buffer(persistent=False)`, instead of a nn.Parameter (because we do not want to learn these fixed cosine and sine values). The exact same rotation process we did for our q^(i) is then done for k^(j), rotating by the corresponding R^j. Notice that this layer has no learnable parameters.

**Problem (rope): Implement RoPE (2 points)**

Deliverable: Implement a class RotaryPositionalEmbedding that applies RoPE to the input tensor. The following interface is recommended:

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    """Construct the RoPE module and create buffers if needed.
    
    theta: float - Θ value for the RoPE
    d_k: int - dimension of query and key vectors
    max_seq_len: int - Maximum sequence length that will be inputted
    device: torch.device | None = None - Device to store the buffer on
    """

def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    """Process an input tensor of shape (..., seq_len, d_k) and return a tensor of the same shape.
    
    Note that you should tolerate x with an arbitrary number of batch dimensions. You should
    assume that the token positions are a tensor of shape (..., seq_len) specifying the token
    positions of x along the sequence dimension.
    
    You should use the token positions to slice your (possibly precomputed) cos and sin tensors
    along the sequence dimension.
    """
```

To test your implementation, complete [adapters.run_rope] and make sure it passes `uv run pytest -k test_rope`.

### 3.5.4 Scaled Dot-Product Attention

We will now implement scaled dot-product attention as described in Vaswani et al. [2017] (section 3.2.1). As a preliminary step, the definition of the Attention operation will make use of softmax, an operation that takes an unnormalized vector of scores and turns it into a normalized distribution:

```
softmax(v)_i = exp(v_i) / Σ^n_{j=1} exp(v_j)                           (10)
```

Note that exp(v_i) can become inf for large values (then, inf/inf = NaN). We can avoid this by noticing that the softmax operation is invariant to adding any constant c to all inputs. We can leverage this property for numerical stability—typically, we will subtract the largest entry of o_i from all elements of o_i, making the new largest entry 0. You will now implement softmax, using this trick for numerical stability.

**Problem (softmax): Implement softmax (1 point)**

Deliverable: Write a function to apply the softmax operation on a tensor. Your function should take two parameters: a tensor and a dimension i, and apply softmax to the i-th dimension of the input tensor. The output tensor should have the same shape as the input tensor, but its i-th dimension will now have a normalized probability distribution. Use the trick of subtracting the maximum value in the i-th dimension from all elements of the i-th dimension to avoid numerical stability issues.

To test your implementation, complete [adapters.run_softmax] and make sure it passes `uv run pytest -k test_softmax_matches_pytorch`.

We can now define the Attention operation mathematically as follows:

```
Attention(Q, K, V) = softmax(Q⊤K/√d_k)V                                (11)
```

where Q ∈ R^{n×d_k}, K ∈ R^{m×d_k}, and V ∈ R^{m×d_v}. Here, Q, K and V are all inputs to this operation—note that these are not the learnable parameters. If you're wondering why this isn't QK⊤, see 3.3.1.

**Masking**: It is sometimes convenient to mask the output of an attention operation. A mask should have the shape M ∈ {True, False}^{n×m}, and each row i of this boolean matrix indicates which keys the query i should attend to. Canonically (and slightly confusingly), a value of True at position (i, j) indicates that the query i does attend to the key j, and a value of False indicates that the query does not attend to the key. In other words, "information flows" at (i, j) pairs with value True. For example, consider a 1 × 3 mask matrix with entries [[True, True, False]]. The single query vector attends only to the first two keys.

Computationally, it will be much more efficient to use masking than to compute attention on subsequences, and we can do this by taking the pre-softmax values (Q⊤K/√d_k) and adding a -∞ in any entry of the mask matrix that is False.

**Problem (scaled_dot_product_attention): Implement scaled dot-product attention (5 points)**

Deliverable: Implement the scaled dot-product attention function. Your implementation should handle keys and queries of shape `(batch_size, ..., seq_len, d_k)` and values of shape `(batch_size, ..., seq_len, d_v)`, where ... represents any number of other batch-like dimensions (if provided). The implementation should return an output with the shape `(batch_size, ..., d_v)`. See section 3.3 for a discussion on batch-like dimensions.

Your implementation should also support an optional user-provided boolean mask of shape `(seq_len, seq_len)`. The attention probabilities of positions with a mask value of True should collectively sum to 1, and the attention probabilities of positions with a mask value of False should be zero.

To test your implementation against our provided tests, you will need to implement the test adapter at [adapters.run_scaled_dot_product_attention]. `uv run pytest -k test_scaled_dot_product_attention` tests your implementation on third-order input tensors, while `uv run pytest -k test_4d_scaled_dot_product_attention` tests your implementation on fourth-order input tensors.

### 3.5.5 Causal Multi-Head Self-Attention

We will implement multi-head self-attention as described in section 3.2.2 of Vaswani et al. [2017]. Recall that, mathematically, the operation of applying multi-head attention is defined as follows:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)                       (12)
```

for head_i = Attention(Q_i, K_i, V_i)                                   (13)

with Q_i, K_i, V_i being slice number i ∈ {1, ..., h} of size d_k or d_v of the embedding dimension for Q, K, and V respectively. With Attention being the scaled dot-product attention operation defined in §3.5.4. From this we can form the multi-head self-attention operation:

```
MultiHeadSelfAttention(x) = W_O MultiHead(W_Q x, W_K x, W_V x)         (14)
```

Here, the learnable parameters are W_Q ∈ R^{hd_k×d_model}, W_K ∈ R^{hd_k×d_model}, W_V ∈ R^{hd_v×d_model}, and W_O ∈ R^{d_model×hd_v}. Since the Qs, K, and Vs are sliced in the multi-head attention operation, we can think of W_Q, W_K and W_V as being separated for each head along the output dimension. When you have this working, you should be computing the key, value, and query projections in a total of three matrix multiplies.

**Causal masking**: Your implementation should prevent the model from attending to future tokens in the sequence. In other words, if the model is given a token sequence t_1, ..., t_n, and we want to calculate the next-word predictions for the prefix t_1, ..., t_i (where i < n), the model should not be able to access (attend to) the token representations at positions t_{i+1}, ..., t_n since it will not have access to these tokens when generating text during inference (and these future tokens leak information about the identity of the true next word, trivializing the language modeling pre-training objective). For an input token sequence t_1, ..., t_n we can naively prevent access to future tokens by running multi-head self-attention n times (for the n unique prefixes in the sequence). Instead, we'll use causal attention masking, which allows token i to attend to all positions j ≤ i in the sequence. You can use `torch.triu` or a broadcasted index comparison to construct this mask, and you should take advantage of the fact that your scaled dot-product attention implementation from §3.5.4 already supports attention masking.

**Applying RoPE**: RoPE should be applied to the query and key vectors, but not the value vectors. Also, the head dimension should be handled as a batch dimension, because in multi-head attention, attention is being applied independently for each head. This means that precisely the same RoPE rotation should be applied to the query and key vectors for each head.

**Problem (multihead_self_attention): Implement causal multi-head self-attention (5 points)**

Deliverable: Implement causal multi-head self-attention as a `torch.nn.Module`. Your implementation should accept (at least) the following parameters:

- `d_model: int` - Dimensionality of the Transformer block inputs.
- `num_heads: int` - Number of heads to use in multi-head self-attention.

Following Vaswani et al. [2017], set d_k = d_v = d_model/h. To test your implementation against our provided tests, implement the test adapter at [adapters.run_multihead_self_attention]. Then, run `uv run pytest -k test_multihead_self_attention` to test your implementation.

## 3.6 The Full Transformer LM

Let's begin by assembling the Transformer block (it will be helpful to refer back to Figure 2). A Transformer block contains two 'sublayers', one for the multihead self attention, and another for the feed-forward network. In each sublayer, we first perform RMSNorm, then the main operation (MHA/FF), finally adding in the residual connection.

To be concrete, the first half (the first 'sub-layer') of the Transformer block should be implementing the following set of updates to produce an output y from an input x,

```
y = x + MultiHeadSelfAttention(RMSNorm(x))                              (15)
```

**Problem (transformer_block): Implement the Transformer block (3 points)**

Implement the pre-norm Transformer block as described in §3.5 and illustrated in Figure 2. Your Transformer block should accept (at least) the following parameters:

- `d_model: int` - Dimensionality of the Transformer block inputs.
- `num_heads: int` - Number of heads to use in multi-head self-attention.
- `d_ff: int` - Dimensionality of the position-wise feed-forward inner layer.

To test your implementation, implement the adapter [adapters.run_transformer_block]. Then run `uv run pytest -k test_transformer_block` to test your implementation.

Deliverable: Transformer block code that passes the provided tests.

Now we put the blocks together, following the high level diagram in Figure 1. Follow our description of the embedding in Section 3.1.1, feed this into `num_layers` Transformer blocks, and then pass that into the three output layers to obtain a distribution over the vocabulary.

**Problem (transformer_lm): Implementing the Transformer LM (3 points)**

Time to put it all together! Implement the Transformer language model as described in §3.1 and illustrated in Figure 1. At minimum, your implementation should accept all the aforementioned construction parameters for the Transformer block, as well as these additional parameters:

- `vocab_size: int` - The size of the vocabulary, necessary for determining the dimensionality of the token embedding matrix.
- `context_length: int` - The maximum context length, necessary for determining the dimensionality of the position embedding matrix.
- `num_layers: int` - The number of Transformer blocks to use.

To test your implementation against our provided tests, you will first need to implement the test adapter at [adapters.run_transformer_lm]. Then, run `uv run pytest -k test_transformer_lm` to test your implementation.

Deliverable: A Transformer LM module that passes the above tests.

**Resource accounting**: It is useful to be able to understand how the various parts of the Transformer consume compute and memory. We will go through the steps to do some basic "FLOPs accounting." The vast majority of FLOPS in a Transformer are matrix multiplies, so our core approach is simple:

1. Write down all the matrix multiplies in a Transformer forward pass.
2. Convert each matrix multiply into FLOPs required.

For this second step, the following fact will be useful:

**Rule**: Given A ∈ R^{m×n} and B ∈ R^{n×p}, the matrix-matrix product AB requires 2mnp FLOPs.

To see this, note that (AB)[i, j] = A[i, :] · B[:, j], and that this dot product requires n additions and n multiplications (2n FLOPs). Then, since the matrix-matrix product AB has m×p entries, the total number of FLOPS is (2n)(mp) = 2mnp.

Now, before you do the next problem, it can be helpful to go through each component of your Transformer block and Transformer LM, and list out all the matrix multiplies and their associated FLOPs costs.

**Problem (transformer_accounting): Transformer LM resource accounting (5 points)**

(a) Consider GPT-2 XL, which has the following configuration:
   - vocab_size: 50,257
   - context_length: 1,024
   - num_layers: 48
   - d_model: 1,600
   - num_heads: 25
   - d_ff: 6,400

   Suppose we constructed our model using this configuration. How many trainable parameters would our model have? Assuming each parameter is represented using single-precision floating point, how much memory is required to just load this model?

   Deliverable: A one-to-two sentence response.

(b) Identify the matrix multiplies required to complete a forward pass of our GPT-2 XL-shaped model. How many FLOPs do these matrix multiplies require in total? Assume that our input sequence has `context_length` tokens.

   Deliverable: A list of matrix multiplies (with descriptions), and the total number of FLOPs required.

(c) Based on your analysis above, which parts of the model require the most FLOPs?

   Deliverable: A one-to-two sentence response.

(d) Repeat your analysis with GPT-2 small (12 layers, 768 d_model, 12 heads), GPT-2 medium (24 layers, 1024 d_model, 16 heads), and GPT-2 large (36 layers, 1280 d_model, 20 heads). As the model size increases, which parts of the Transformer LM take up proportionally more or less of the total FLOPs?

   Deliverable: For each model, provide a breakdown of model components and its associated FLOPs (as a proportion of the total FLOPs required for a forward pass). In addition, provide a one-to-two sentence description of how varying the model size changes the proportional FLOPs of each component.

(e) Take GPT-2 XL and increase the context length to 16,384. How does the total FLOPs for one forward pass change? How do the relative contribution of FLOPs of the model components change?

   Deliverable: A one-to-two sentence response.