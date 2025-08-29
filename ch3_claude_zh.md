# 3 Transformer语言模型架构

语言模型接受批量的整数标记ID序列作为输入（即形状为 `(batch_size, sequence_length)` 的 `torch.Tensor`），并返回词表上的（批量）归一化概率分布（即形状为 `(batch_size, sequence_length, vocab_size)` 的 PyTorch 张量），其中预测分布是对每个输入标记的下一个词的预测。在训练语言模型时，我们使用这些下一个词的预测来计算实际下一个词和预测下一个词之间的交叉熵损失。在推理期间从语言模型生成文本时，我们从最后一个时间步（即序列中的最后一项）获取预测的下一个词分布来生成序列中的下一个标记（例如，通过选择概率最高的标记、从分布中采样等），将生成的标记添加到输入序列中，并重复此过程。

在作业的这一部分，您将从头开始构建这个 Transformer 语言模型。我们将首先从模型的高级描述开始，然后逐步详细描述各个组件。

## 3.1 Transformer 语言模型

给定一个标记ID序列，Transformer语言模型使用输入嵌入将标记ID转换为稠密向量，通过 `num_layers` 个Transformer块传递嵌入的标记，然后应用学习的线性投影（"输出嵌入"或"LM头"）来产生预测的下一个标记logits。参见图1的示意图表示。

![Transformer LM Overview](figure1-placeholder)
*图1: 我们的Transformer语言模型概述。*

### 3.1.1 标记嵌入

在第一步中，Transformer将标记ID的（批量）序列嵌入到包含标记身份信息的向量序列中（图1中的红色块）。

更具体地说，给定一个标记ID序列，Transformer语言模型使用标记嵌入层来产生向量序列。每个嵌入层接收形状为 `(batch_size, sequence_length)` 的整数张量，并产生形状为 `(batch_size, sequence_length, d_model)` 的向量序列。

### 3.1.2 Pre-norm Transformer块

在嵌入之后，激活由几个结构相同的神经网络层处理。标准的仅解码器Transformer语言模型由 `num_layers` 个相同的层（通常称为Transformer"块"）组成。每个Transformer块接收形状为 `(batch_size, sequence_length, d_model)` 的输入并返回形状为 `(batch_size, sequence_length, d_model)` 的输出。每个块跨序列聚合信息（通过自注意力）并非线性地转换它（通过前馈层）。

## 3.2 输出归一化和嵌入

在 `num_layers` 个Transformer块之后，我们将采用最终激活并将其转换为词表上的分布。

我们将实现"pre-norm"Transformer块（在§3.5中详述），它还需要在最终Transformer块之后使用层归一化（如下所述）以确保其输出正确缩放。

在此归一化之后，我们将使用标准的学习线性变换将Transformer块的输出转换为预测的下一个标记logits（参见，例如，Radford et al. [2018] 方程2）。

## 3.3 备注：批处理、Einsum和高效计算

在整个Transformer中，我们将对许多批量输入执行相同的计算。以下是几个例子：

- **批次元素**：我们对每个批次元素应用相同的Transformer前向操作。
- **序列长度**：像RMSNorm和前馈这样的"位置方式"操作在序列的每个位置上都相同地操作。
- **注意力头**：注意力操作在"多头"注意力操作中跨注意力头批处理。

以充分利用GPU的方式执行此类操作并且易于阅读和理解是很有用的。许多PyTorch操作可以在张量开始处接受多余的"批量"维度，并在这些维度上高效地重复/广播操作。

例如，假设我们正在进行位置方式的批处理操作。我们有一个形状为 `(batch_size, sequence_length, d_model)` 的"数据张量"D，我们想对形状为 `(d_model, d_model)` 的矩阵A进行批量向量-矩阵乘法。在这种情况下，`D @ A` 将进行批量矩阵乘法，这是PyTorch中的高效原语，其中 `(batch_size, sequence_length)` 维度被批处理。

因此，假设您的函数可能被给予额外的批量维度并将这些维度保持在PyTorch形状的开始处是有帮助的。为了组织张量以便它们能够以这种方式批处理，它们可能需要使用许多步骤的view、reshape和transpose来形成。这可能有点痛苦，并且通常很难阅读代码在做什么以及张量的形状是什么。

一个更符合人体工程学的选择是在 `torch.einsum` 内使用einsum符号，或者更准确地说使用框架无关的库，如 `einops` 或 `einx`。两个关键操作是 `einsum`，它可以对具有任意输入张量维度的张量收缩进行处理，以及 `rearrange`，它可以重新排序、连接和拆分任意维度。事实证明，机器学习中几乎所有操作都是维度调整和张量收缩以及偶尔的（通常是点方式的）非线性函数的某种组合。这意味着当使用einsum符号时，您的很多代码可以更具可读性和灵活性。

我们强烈建议在课程中学习和使用einsum符号。之前没有接触过einsum符号的学生应该使用 `einops`（文档在这里），已经熟悉 `einops` 的学生应该学习更通用的 `einx`（这里）。我们提供的环境中已经安装了这两个包。

这里我们给出一些如何使用einsum符号的例子。这些是einops文档的补充，您应该首先阅读该文档。

### 示例：使用einops.einsum进行批量矩阵乘法

```python
import torch
from einops import rearrange, einsum

## 基本实现
Y = D @ A.T
# 很难说明输入和输出形状以及它们的含义。
# D和A可以有什么形状，这些是否有意外行为？

## Einsum是自文档化和健壮的
# D A -> Y
Y = einsum(D, A, "batch sequence d_in, d_out d_in -> batch sequence d_out")

## 或者，一个批量版本，其中D可以有任何前导维度，但A受约束。
Y = einsum(D, A, "... d_in, d_out d_in -> ... d_out")
```

### 示例：使用einops.rearrange进行广播操作

我们有一批图像，对于每个图像，我们想要基于某个缩放因子生成10个调暗版本：

```python
images = torch.randn(64, 128, 128, 3) # (batch, height, width, channel)
dim_by = torch.linspace(start=0.0, end=1.0, steps=10)

## 重塑和乘法
dim_value = rearrange(dim_by, "dim_value -> 1 dim_value 1 1 1")
images_rearr = rearrange(images, "b height width channel -> b 1 height width channel")
dimmed_images = images_rearr * dim_value

## 或一次性完成：
dimmed_images = einsum(
    images, dim_by,
    "batch height width channel, dim_value -> batch dim_value height width channel"
)
```

### 示例：使用einops.rearrange进行像素混合

假设我们有一批图像，表示为形状为 `(batch, height, width, channel)` 的张量，我们想要在图像的所有像素上执行线性变换，但这种变换应该独立地为每个通道进行。我们的线性变换表示为形状为 `(height × width, height × width)` 的矩阵B。

```python
channels_last = torch.randn(64, 32, 32, 3) # (batch, height, width, channel)
B = torch.randn(32*32, 32*32)

## 重新排列图像张量以便在所有像素上进行混合
channels_last_flat = channels_last.view(
    -1, channels_last.size(1) * channels_last.size(2), channels_last.size(3)
)
channels_first_flat = channels_last_flat.transpose(1, 2)
channels_first_flat_transformed = channels_first_flat @ B.T
channels_last_flat_transformed = channels_first_flat_transformed.transpose(1, 2)
channels_last_transformed = channels_last_flat_transformed.view(*channels_last.shape)
```

相反，使用einops：

```python
height = width = 32
## Rearrange替换笨拙的torch view + transpose
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

或者，如果你感觉疯狂：使用 `einx.dot`（einx等效于 `einops.einsum`）一次性完成所有操作：

```python
height = width = 32
channels_last_transformed = einx.dot(
    "batch row_in col_in channel, (row_out col_out) (row_in col_in)"
    "-> batch row_out col_out channel",
    channels_last, B,
    col_in=width, col_out=width
)
```

这里的第一个实现可以通过在之前和之后放置注释来改进，以指示输入和输出形状是什么，但这很笨拙且容易出错。使用einsum符号，文档就是实现！

Einsum符号可以处理任意输入批处理维度，但也有自我文档化的关键优势。在使用einsum符号的代码中，输入和输出张量的相关形状更加清晰。对于其余张量，您可以考虑使用张量类型提示，例如使用 `jaxtyping` 库（不特定于Jax）。

我们将在作业2中更多地讨论使用einsum符号的性能影响，但现在知道它们几乎总是比替代方案更好！

### 3.3.1 数学符号和内存排序

许多机器学习论文在其符号中使用行向量，这导致与NumPy和PyTorch默认使用的行主内存排序很好配合的表示。使用行向量，线性变换看起来像

```
y = xW⊤                                                                  (1)
```

对于行主 W ∈ R^(d_out×d_in) 和行向量 x ∈ R^(1×d_in)。

在线性代数中，通常更常见的是使用列向量，其中线性变换看起来像

```
y = Wx                                                                   (2)
```

给定行主 W ∈ R^(d_out×d_in) 和列向量 x ∈ R^d_in。我们将在此作业中使用列向量进行数学符号，因为这样通常更容易理解数学。您应该记住，如果您想使用普通矩阵乘法符号，您将必须使用行向量约定应用矩阵，因为PyTorch使用行主内存排序。如果您对矩阵操作使用einsum，这应该不是问题。

## 3.4 基本构建块：线性和嵌入模块

### 3.4.1 参数初始化

有效训练神经网络通常需要仔细初始化模型参数——糟糕的初始化可能导致不良行为，如梯度消失或爆炸。Pre-norm transformers对初始化异常健壮，但它们仍然可能对训练速度和收敛性产生重大影响。由于这个作业已经很长，我们将把细节保存到作业3，而是给您一些应该在大多数情况下都有效的近似初始化。现在，使用：

- **线性权重**：N(μ = 0, σ² = 2/(d_in+d_out)) 在 [-3σ, 3σ] 截断。
- **嵌入**：N(μ = 0, σ² = 1) 在 [-3, 3] 截断
- **RMSNorm**：1

您应该使用 `torch.nn.init.trunc_normal_` 来初始化截断正态权重。

### 3.4.2 线性模块

线性层是Transformers和一般神经网络的基本构建块。首先，您将实现从 `torch.nn.Module` 继承的自己的线性类，并执行线性变换：

```
y = Wx                                                                   (3)
```

请注意，我们不包括偏置项，遵循大多数现代LLM。

**问题（linear）：实现线性模块（1分）**

交付物：实现从 `torch.nn.Module` 继承并执行线性变换的线性类。您的实现应该遵循PyTorch内置 `nn.Linear` 模块的接口，除了没有bias参数或参数。我们推荐以下接口：

```python
def __init__(self, in_features, out_features, device=None, dtype=None):
    """构造线性变换模块。此函数应接受以下参数：
    
    in_features: int - 输入的最终维度
    out_features: int - 输出的最终维度
    device: torch.device | None = None - 存储参数的设备
    dtype: torch.dtype | None = None - 参数的数据类型
    """

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """将线性变换应用于输入。"""
```

确保：
- 子类化nn.Module
- 调用超类构造函数
- 构造并存储您的参数为W（不是W⊤）出于内存排序原因，将其放在nn.Parameter中
- 当然，不要使用nn.Linear或nn.functional.linear

对于初始化，使用上面的设置以及 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试您的线性模块，在[adapters.run_linear]处实现测试适配器。适配器应该将给定的权重加载到您的线性模块中。您可以使用Module.load_state_dict来实现此目的。然后，运行 `uv run pytest -k test_linear`。

### 3.4.3 嵌入模块

如上所述，Transformer的第一层是将整数标记ID映射到维度为d_model的向量空间的嵌入层。我们将实现从 `torch.nn.Module` 继承的自定义嵌入类（所以您不应该使用nn.Embedding）。前向方法应该通过使用形状为 `(batch_size, sequence_length)` 的标记ID的 `torch.LongTensor` 索引形状为 `(vocab_size, d_model)` 的嵌入矩阵来选择每个标记ID的嵌入向量。

**问题（embedding）：实现嵌入模块（1分）**

交付物：实现从 `torch.nn.Module` 继承并执行嵌入查找的嵌入类。您的实现应该遵循PyTorch内置 `nn.Embedding` 模块的接口。我们推荐以下接口：

```python
def __init__(self, num_embeddings, embedding_dim, device=None, dtype=None):
    """构造嵌入模块。此函数应接受以下参数：
    
    num_embeddings: int - 词表大小
    embedding_dim: int - 嵌入向量的维度，即d_model
    device: torch.device | None = None - 存储参数的设备
    dtype: torch.dtype | None = None - 参数的数据类型
    """

def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
    """查找给定标记ID的嵌入向量。"""
```

确保：
- 子类化nn.Module
- 调用超类构造函数
- 将您的嵌入矩阵初始化为nn.Parameter
- 存储嵌入矩阵，其中d_model是最终维度
- 当然，不要使用nn.Embedding或nn.functional.embedding

再次，使用上面的初始化设置，并使用 `torch.nn.init.trunc_normal_` 来初始化权重。

要测试您的实现，在[adapters.run_embedding]处实现测试适配器。然后，运行 `uv run pytest -k test_embedding`。

## 3.5 Pre-Norm Transformer块

每个Transformer块有两个子层：多头自注意力机制和位置前馈网络（Vaswani et al., 2017，第3.1节）。

在原始Transformer论文中，模型在两个子层中的每一个周围使用残差连接，然后进行层归一化。这种架构通常称为"post-norm"Transformer，因为层归一化应用于子层输出。然而，各种工作发现将层归一化从每个子层的输出移动到每个子层的输入（在最终Transformer块之后附加层归一化）改善了Transformer训练稳定性[Nguyen和Salazar，2019，Xiong等人，2020]——参见图2中此"pre-norm"Transformer块的可视化表示。然后通过残差连接将每个Transformer块子层的输出添加到子层输入中（Vaswani等人，2017，第5.4节）。pre-norm的直觉是从输入嵌入到Transformer的最终输出有一个没有任何归一化的干净"残差流"，据说可以改善梯度流。这个pre-norm Transformer现在是当今语言模型中使用的标准（例如，GPT-3，LLaMA，PaLM等），所以我们将实现这个变体。我们将逐步介绍pre-norm Transformer块的每个组件，按顺序实现它们。

![Pre-norm Transformer Block](figure2-placeholder)
*图2: pre-norm Transformer块。*

### 3.5.1 均方根层归一化

Vaswani等人[2017]的原始Transformer实现使用层归一化[Ba等人，2016]来归一化激活。遵循Touvron等人[2023]，我们将使用均方根层归一化（RMSNorm；Zhang和Sennrich，2019，方程4）进行层归一化。给定激活向量 a ∈ R^d_model，RMSNorm将按如下方式重新缩放每个激活 a_i：

```
RMSNorm(a_i) = (a_i / RMS(a)) * g_i                                     (4)
```

其中 `RMS(a) = √(1/d_model * Σ(a_i²) + ε)`。这里，g_i 是可学习的"增益"参数（总共有 d_model 个这样的参数），ε 是通常固定为 1e-5 的超参数。

您应该将输入向上转换为 `torch.float32` 以防止在输入平方时溢出。总体上，您的前向方法应该看起来像：

```python
in_dtype = x.dtype
x = x.to(torch.float32)
# 您的代码在这里执行RMSNorm
...
result = ...
# 以原始dtype返回结果
return result.to(in_dtype)
```

**问题（rmsnorm）：均方根层归一化（1分）**

交付物：将RMSNorm实现为 `torch.nn.Module`。我们推荐以下接口：

```python
def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
    """构造RMSNorm模块。此函数应接受以下参数：
    
    d_model: int - 模型的隐藏维度
    eps: float = 1e-5 - 数值稳定性的Epsilon值
    device: torch.device | None = None - 存储参数的设备
    dtype: torch.dtype | None = None - 参数的数据类型
    """

def forward(self, x: torch.Tensor) -> torch.Tensor:
    """处理形状为(batch_size, sequence_length, d_model)的输入张量
    并返回相同形状的张量。"""
```

注意：记住在执行归一化之前将输入向上转换为 `torch.float32`（然后向下转换为原始dtype），如上所述。

要测试您的实现，在[adapters.run_rmsnorm]处实现测试适配器。然后，运行 `uv run pytest -k test_rmsnorm`。

### 3.5.2 位置前馈网络

![SiLU vs ReLU](figure3-placeholder)
*图3: 比较SiLU（又名Swish）和ReLU激活函数。*

在原始Transformer论文（Vaswani等人[2017]的第3.3节）中，Transformer前馈网络由两个线性变换组成，中间有一个ReLU激活（ReLU(x) = max(0, x)）。内部前馈层的维度通常是输入维度的4倍。

然而，现代语言模型倾向于与这个原始设计相比纳入两个主要变化：它们使用另一个激活函数并采用门控机制。具体而言，我们将实现在Llama 3 [Grattafiori等人，2024]和Qwen 2.5 [Yang等人，2024]等LLM中采用的"SwiGLU"激活函数，它将SiLU（通常称为Swish）激活与称为门控线性单元（GLU）的门控机制相结合。我们还将省略有时在线性层中使用的偏置项，遵循自PaLM [Chowdhery等人，2022]和LLaMA [Touvron等人，2023]以来的大多数现代LLM。

SiLU或Swish激活函数[Hendrycks和Gimpel，2016，Elfwing等人，2017]定义如下：

```
SiLU(x) = x · σ(x) = x / (1 + e^(-x))                                   (5)
```

如图3所示，SiLU激活函数类似于ReLU激活函数，但在零处是平滑的。

门控线性单元（GLU）最初由Dauphin等人[2017]定义为通过sigmoid函数传递的线性变换与另一个线性变换的元素乘积：

```
GLU(x, W1, W2) = σ(W1x) ⊙ W2x                                          (6)
```

其中⊙表示元素乘法。门控线性单元被建议"通过为梯度提供线性路径同时保持非线性能力来减少深度架构的梯度消失问题。"

将SiLU/Swish和GLU结合在一起，我们得到SwiGLU，我们将其用于前馈网络：

```
FFN(x) = SwiGLU(x, W1, W2, W3) = W2(SiLU(W1x) ⊙ W3x)                  (7)
```

其中 x ∈ R^d_model，W1, W3 ∈ R^(d_ff×d_model)，W2 ∈ R^(d_model×d_ff)，规范地，d_ff = (8/3)d_model。

Shazeer [2020]首先提出将SiLU/Swish激活与GLU结合，并进行了实验，表明SwiGLU在语言建模任务上优于ReLU和SiLU（无门控）等基线。稍后在作业中，您将比较SwiGLU和SiLU。虽然我们已经提到了这些组件的一些启发式论证（论文提供了更多支持证据），但保持经验视角是好的：Shazeer论文中现在著名的引用是

> 我们对这些架构似乎有效的原因不提供任何解释；我们将它们的成功归因于神圣的仁慈，如同所有其他事物一样。

**问题（positionwise_feedforward）：实现位置前馈网络（2分）**

交付物：实现由SiLU激活函数和GLU组成的SwiGLU前馈网络。

注意：在这种特定情况下，您应该可以在实现中使用 `torch.sigmoid` 以获得数值稳定性。

您应该在实现中将d_ff设置为约(8/3) × d_model，同时确保内部前馈层的维度是64的倍数，以充分利用您的硬件。要根据我们提供的测试测试您的实现，您需要在[adapters.run_swiglu]处实现测试适配器。然后，运行 `uv run pytest -k test_swiglu` 来测试您的实现。

### 3.5.3 相对位置嵌入

为了向模型注入位置信息，我们将实现旋转位置嵌入[Su等人，2021]，通常称为RoPE。对于标记位置i处的给定查询标记 q^(i) = W_q x^(i) ∈ R^d，我们将应用成对旋转矩阵 R_i，给出 q'^(i) = R_i q^(i) = R_i W_q x^(i)。这里，R_i 将嵌入元素对 q^(i)_{2k-1:2k} 作为2d向量按角度 θ_{i,k} = i/Θ^{(2k-1)/d} 旋转，对于 k ∈ {1, ..., d/2} 和某个常数 Θ。因此，我们可以考虑 R_i 为大小为 d × d 的块对角矩阵，对于 k ∈ {1, ..., d/2}，有块 R^i_k，其中

```
R^i_k = [cos(θ_{i,k})  -sin(θ_{i,k})]                                   (8)
        [sin(θ_{i,k})   cos(θ_{i,k})]
```

因此我们得到完整的旋转矩阵

```
R^i = [R^i_1   0      0    ...  0    ]                                  (9)
      [0       R^i_2   0    ...  0    ]
      [0       0       R^i_3 ...  0    ]
      [...     ...     ... ...   ...  ]
      [0       0       0    ... R^i_{d/2}]
```

其中0表示2 × 2零矩阵。虽然可以构造完整的d × d矩阵，但好的解决方案应该使用此矩阵的属性来更高效地实现变换。由于我们只关心给定序列内标记的相对旋转，我们可以在层之间和不同批次之间重用我们为cos(θ_{i,k})和sin(θ_{i,k})计算的值。如果您想优化它，您可以使用所有层引用的单个RoPE模块，它可以有一个在init期间使用 `self.register_buffer(persistent=False)` 创建的sin和cos值的2d预计算缓冲区，而不是nn.Parameter（因为我们不想学习这些固定的余弦和正弦值）。然后对k^(j)执行与我们对q^(i)相同的旋转过程，通过相应的R^j旋转。注意，此层没有可学习参数。

**问题（rope）：实现RoPE（2分）**

交付物：实现一个RotaryPositionalEmbedding类，将RoPE应用于输入张量。推荐以下接口：

```python
def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
    """构造RoPE模块并在需要时创建缓冲区。
    
    theta: float - RoPE的Θ值
    d_k: int - 查询和键向量的维度
    max_seq_len: int - 将要输入的最大序列长度
    device: torch.device | None = None - 存储缓冲区的设备
    """

def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
    """处理形状为(..., seq_len, d_k)的输入张量并返回相同形状的张量。
    
    注意您应该容忍具有任意数量批次维度的x。您应该
    假设标记位置是形状为(..., seq_len)的张量，指定x沿序列维度的标记
    位置。
    
    您应该使用标记位置沿序列维度切片您的（可能预计算的）cos和sin张量。
    """
```

要测试您的实现，完成[adapters.run_rope]并确保它通过 `uv run pytest -k test_rope`。

### 3.5.4 缩放点积注意力

我们现在将实现Vaswani等人[2017]（第3.2.1节）中描述的缩放点积注意力。作为初步步骤，注意力操作的定义将使用softmax，这是一个将未归一化分数向量转化为归一化分布的操作：

```
softmax(v)_i = exp(v_i) / Σ^n_{j=1} exp(v_j)                           (10)
```

注意，对于大值，exp(v_i)可能变为inf（然后，inf/inf = NaN）。我们可以通过注意到softmax操作对向所有输入添加任何常数c是不变的来避免这一点。我们可以利用这个属性来获得数值稳定性——通常，我们将从o_i的所有元素中减去o_i的最大条目，使新的最大条目为0。您现在将实现softmax，使用此技巧获得数值稳定性。

**问题（softmax）：实现softmax（1分）**

交付物：编写一个函数来对张量应用softmax操作。您的函数应该接受两个参数：一个张量和一个维度i，并对输入张量的第i个维度应用softmax。输出张量应该具有与输入张量相同的形状，但其第i个维度现在将具有归一化概率分布。使用从第i个维度的所有元素中减去第i个维度中的最大值的技巧来避免数值稳定性问题。

要测试您的实现，完成[adapters.run_softmax]并确保它通过 `uv run pytest -k test_softmax_matches_pytorch`。

我们现在可以从数学角度定义注意力操作如下：

```
Attention(Q, K, V) = softmax(Q⊤K/√d_k)V                                (11)
```

其中 Q ∈ R^{n×d_k}，K ∈ R^{m×d_k}，V ∈ R^{m×d_v}。这里，Q、K和V都是此操作的输入——注意这些不是可学习参数。如果您想知道为什么这不是QK⊤，请参见3.3.1。

**掩码**：有时掩码注意力操作的输出是方便的。掩码应该具有形状 M ∈ {True, False}^{n×m}，这个布尔矩阵的每一行i指示查询i应该关注哪些键。规范地（稍有混淆），位置(i, j)处的True值表示查询i确实关注键j，False值表示查询不关注键。换句话说，"信息流"在值为True的(i, j)对处。例如，考虑具有条目[[True, True, False]]的1 × 3掩码矩阵。单个查询向量只关注前两个键。

计算上，使用掩码比计算子序列上的注意力要高效得多，我们可以通过获取pre-softmax值(Q⊤K/√d_k)并在掩码矩阵为False的任何条目中添加-∞来做到这一点。

**问题（scaled_dot_product_attention）：实现缩放点积注意力（5分）**

交付物：实现缩放点积注意力函数。您的实现应该处理形状为 `(batch_size, ..., seq_len, d_k)` 的键和查询以及形状为 `(batch_size, ..., seq_len, d_v)` 的值，其中...表示任何数量的其他批量维度（如果提供）。实现应该返回形状为 `(batch_size, ..., d_v)` 的输出。有关批量维度的讨论，请参见第3.3节。

您的实现还应该支持形状为 `(seq_len, seq_len)` 的可选用户提供的布尔掩码。具有掩码值True的位置的注意力概率应该总和为1，具有掩码值False的位置的注意力概率应该为零。

要根据我们提供的测试测试您的实现，您需要在[adapters.run_scaled_dot_product_attention]处实现测试适配器。`uv run pytest -k test_scaled_dot_product_attention` 在三阶输入张量上测试您的实现，而 `uv run pytest -k test_4d_scaled_dot_product_attention` 在四阶输入张量上测试您的实现。

### 3.5.5 因果多头自注意力

我们将实现Vaswani等人[2017]第3.2.2节中描述的多头自注意力。回想一下，从数学角度来说，应用多头注意力的操作定义如下：

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)                       (12)
```

对于 head_i = Attention(Q_i, K_i, V_i)                                   (13)

其中 Q_i, K_i, V_i 是Q、K和V的嵌入维度的大小为d_k或d_v的切片编号i ∈ {1, ..., h}。通过注意力是§3.5.4中定义的缩放点积注意力操作。由此我们可以形成多头自注意力操作：

```
MultiHeadSelfAttention(x) = W_O MultiHead(W_Q x, W_K x, W_V x)         (14)
```

这里，可学习参数是 W_Q ∈ R^{hd_k×d_model}，W_K ∈ R^{hd_k×d_model}，W_V ∈ R^{hd_v×d_model}，和 W_O ∈ R^{d_model×hd_v}。由于Q、K和V在多头注意力操作中被切片，我们可以将W_Q、W_K和W_V视为沿输出维度为每个头分离。当您让这个工作时，您应该总共用三次矩阵乘法计算键、值和查询投影。

**因果掩码**：您的实现应该防止模型关注序列中的未来标记。换句话说，如果模型被给予标记序列 t_1, ..., t_n，我们想计算前缀 t_1, ..., t_i（其中 i < n）的下一个词预测，模型不应能够访问（关注）位置 t_{i+1}, ..., t_n 的标记表示，因为在推理期间生成文本时它不会访问这些标记（这些未来标记泄露关于真实下一个词身份的信息，使语言建模预训练目标变得平凡）。对于输入标记序列 t_1, ..., t_n，我们可以天真地通过n次运行多头自注意力（对序列中的n个独特前缀）来防止访问未来标记。相反，我们将使用因果注意力掩码，它允许标记i关注序列中所有位置 j ≤ i。您可以使用 `torch.triu` 或广播索引比较来构造此掩码，您应该利用您的缩放点积注意力实现从§3.5.4已经支持注意力掩码这一事实。

**应用RoPE**：RoPE应该应用于查询和键向量，但不应用于值向量。此外，头维度应该作为批次维度处理，因为在多头注意力中，每个头都独立应用注意力。这意味着对每个头的查询和键向量应该应用完全相同的RoPE旋转。

**问题（multihead_self_attention）：实现因果多头自注意力（5分）**

交付物：将因果多头自注意力实现为 `torch.nn.Module`。您的实现应该接受（至少）以下参数：

- `d_model: int` - Transformer块输入的维度。
- `num_heads: int` - 多头自注意力中使用的头数。

遵循Vaswani等人[2017]，设置 d_k = d_v = d_model/h。要根据我们提供的测试测试您的实现，在[adapters.run_multihead_self_attention]处实现测试适配器。然后，运行 `uv run pytest -k test_multihead_self_attention` 来测试您的实现。

## 3.6 完整的Transformer语言模型

让我们开始组装Transformer块（参考图2会有帮助）。Transformer块包含两个'子层'，一个用于多头自注意力，另一个用于前馈网络。在每个子层中，我们首先执行RMSNorm，然后是主要操作（MHA/FF），最后添加残差连接。

具体来说，Transformer块的前半部分（第一个'子层'）应该实现以下更新集合以从输入x产生输出y，

```
y = x + MultiHeadSelfAttention(RMSNorm(x))                              (15)
```

**问题（transformer_block）：实现Transformer块（3分）**

实现§3.5中描述和图2中说明的pre-norm Transformer块。您的Transformer块应该接受（至少）以下参数：

- `d_model: int` - Transformer块输入的维度。
- `num_heads: int` - 多头自注意力中使用的头数。
- `d_ff: int` - 位置前馈内层的维度。

要测试您的实现，实现适配器[adapters.run_transformer_block]。然后运行 `uv run pytest -k test_transformer_block` 来测试您的实现。

交付物：通过提供的测试的Transformer块代码。

现在我们将块组合在一起，遵循图1中的高级图表。遵循我们在第3.1.1节中对嵌入的描述，将其馈入 `num_layers` 个Transformer块，然后将其传递到三个输出层以获得词表上的分布。

**问题（transformer_lm）：实现Transformer语言模型（3分）**

是时候将所有内容整合在一起了！实现§3.1中描述和图1中说明的Transformer语言模型。至少，您的实现应该接受Transformer块的所有上述构造参数，以及这些额外参数：

- `vocab_size: int` - 词表大小，确定标记嵌入矩阵维度所必需的。
- `context_length: int` - 最大上下文长度，确定位置嵌入矩阵维度所必需的。
- `num_layers: int` - 要使用的Transformer块数量。

要根据我们提供的测试测试您的实现，您首先需要在[adapters.run_transformer_lm]处实现测试适配器。然后，运行 `uv run pytest -k test_transformer_lm` 来测试您的实现。

交付物：通过上述测试的Transformer语言模型模块。

**资源核算**：能够理解Transformer的各个部分如何消耗计算和内存是有用的。我们将经历进行一些基本"FLOPs核算"的步骤。Transformer中绝大部分FLOPS是矩阵乘法，所以我们的核心方法很简单：

1. 写下Transformer前向传递中的所有矩阵乘法。
2. 将每个矩阵乘法转换为所需的FLOPs。

对于第二步，以下事实将是有用的：

**规则**：给定 A ∈ R^{m×n} 和 B ∈ R^{n×p}，矩阵乘积 AB 需要 2mnp FLOPs。

要看到这一点，注意 (AB)[i, j] = A[i, :] · B[:, j]，此点积需要n次加法和n次乘法（2n FLOPs）。然后，由于矩阵乘积AB有m×p个条目，总FLOPs数是(2n)(mp) = 2mnp。

现在，在您做下一个问题之前，查看您的Transformer块和Transformer语言模型的每个组件，列出所有矩阵乘法及其相关FLOPs成本可能会有帮助。

**问题（transformer_accounting）：Transformer语言模型资源核算（5分）**

(a) 考虑GPT-2 XL，它具有以下配置：
   - vocab_size: 50,257
   - context_length: 1,024
   - num_layers: 48
   - d_model: 1,600
   - num_heads: 25
   - d_ff: 6,400

   假设我们使用此配置构建了我们的模型。我们的模型将有多少个可训练参数？假设每个参数使用单精度浮点表示，仅加载此模型需要多少内存？

   交付物：一到两句话的回应。

(b) 识别完成我们的GPT-2 XL形状模型的前向传递所需的矩阵乘法。这些矩阵乘法总共需要多少FLOPs？假设我们的输入序列有 `context_length` 个标记。

   交付物：矩阵乘法列表（带描述）和所需的总FLOPs数。

(c) 基于您上面的分析，模型的哪些部分需要最多的FLOPs？

   交付物：一到两句话的回应。

(d) 用GPT-2 small（12层，768 d_model，12头）、GPT-2 medium（24层，1024 d_model，16头）和GPT-2 large（36层，1280 d_model，20头）重复您的分析。随着模型大小增加，Transformer语言模型的哪些部分在总FLOPs中占比例地更多或更少？

   交付物：对于每个模型，提供模型组件的分解及其相关FLOPs（作为前向传递所需总FLOPs的比例）。此外，提供一到两句关于改变模型大小如何改变每个组件比例FLOPs的描述。

(e) 取GPT-2 XL并将上下文长度增加到16,384。一次前向传递的总FLOPs如何改变？模型组件的FLOPs相对贡献如何改变？

   交付物：一到两句话的回应。