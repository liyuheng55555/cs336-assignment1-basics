import logging
from pathlib import Path

import einops
import torch
import numpy as np

from ch3.Softmax import softmax
from ch3.TransformerLM import TransformerLM
from ch4.cross_entropy import cross_entropy
from ch4.gradient_clipping import gradient_clipping
from ch4.optimizer_AdamW import AdamW
from ch5.checkpoint import save_checkpoint
from ch5.get_batch import get_batch
from tests.test_tokenizer import VOCAB_PATH

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
ROPE_THETA = 10000
NUM_HEADS = 16
NUM_LAYERS = 4

TOTAL_STEPS = 5001

# ADAMW_PARAMS
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01

# GRADIENT_CLIPPING
L2_NORM = 1.0

torch.manual_seed(69)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

weights = {
    'token_embeddings.weight': torch.empty(VOCAB_SIZE, D_MODEL).normal_(mean=0.0, std=0.02),
    'ln_final.weight': torch.rand(D_MODEL),
    'lm_head.weight': torch.rand(VOCAB_SIZE, D_MODEL),
}
for i in range(NUM_LAYERS):
    layer_weights = {
        f'layers.{i}.attn.q_proj.weight': torch.rand(D_MODEL, D_MODEL),
        f'layers.{i}.attn.k_proj.weight': torch.rand(D_MODEL, D_MODEL),
        f'layers.{i}.attn.v_proj.weight': torch.rand(D_MODEL, D_MODEL),
        f'layers.{i}.attn.output_proj.weight': torch.rand(D_MODEL, D_MODEL),
        f'layers.{i}.ffn.w1.weight': torch.rand(D_FF, D_MODEL),
        f'layers.{i}.ffn.w2.weight': torch.rand(D_MODEL, D_FF),
        f'layers.{i}.ffn.w3.weight': torch.rand(D_FF, D_MODEL),
        f'layers.{i}.ln1.weight': torch.rand(D_MODEL),
        f'layers.{i}.ln2.weight': torch.rand(D_MODEL),
    }
    weights |= layer_weights

model = TransformerLM(
    VOCAB_SIZE,
    CONTEXT_LENGTH,
    D_MODEL,
    NUM_LAYERS,
    NUM_HEADS,
    D_FF,
    ROPE_THETA,
    weights
)

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
    eps=EPS
)

# for param in model.parameters():
#     print(param.shape)

for name, param in model.named_parameters():
    print(name, param.shape)

data_path = Path("../ch2/tokenized_tiny_story/result.npy")
data = np.load(data_path, mmap_mode="r")
checkpoint_dir = Path("checkpoints")

for iteration in range(TOTAL_STEPS):
    batch, target = get_batch(data, batch_size=32, context_length=CONTEXT_LENGTH, device="cpu")
    result = model.forward(batch.long())
    result = einops.rearrange(result, "batch context_length vocab_size -> context_length batch vocab_size")
    target = einops.rearrange(target, "batch context_length -> context_length batch")
    entropy: torch.Tensor = cross_entropy(result, target)
    entropy.backward()
    gradient_clipping(model.parameters(), L2_NORM)

    optimizer.step()
    optimizer.zero_grad()

    if iteration % 1000 == 0:
        logging.info("saving checkpoint...")
        ckpt_path = checkpoint_dir/f"{iteration}.ckpt"
        save_checkpoint(model, optimizer, iteration, ckpt_path)
        logging.info(f"checkpoint {ckpt_path.__str__()} saved")
    logging.info(f"iteration: {iteration:06d}loss: {entropy.int()}")


save_checkpoint(model, optimizer, TOTAL_STEPS, checkpoint_dir/f"{TOTAL_STEPS}.ckpt")
