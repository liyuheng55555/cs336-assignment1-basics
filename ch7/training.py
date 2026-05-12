import logging
import time
from pathlib import Path

import einops
import torch
import numpy as np
from jaxtyping import Float
from torch import Tensor

from ch2.Tokenizer import Tokenizer
from ch3.Softmax import softmax
from ch3.TransformerLM import TransformerLM
from ch4.cross_entropy import cross_entropy
from ch4.gradient_clipping import gradient_clipping
from ch4.optimizer_AdamW import AdamW
from ch5.checkpoint import save_checkpoint, load_checkpoint
from ch5.get_batch import get_batch

############## Settings ##############

TOTAL_STEPS = 5001

# Model Size

VOCAB_SIZE = 10000
CONTEXT_LENGTH = 256
D_MODEL = 512
D_FF = 1344
ROPE_THETA = 10000
NUM_HEADS = 16
NUM_LAYERS = 4

# ADAMW_PARAMS
LEARNING_RATE = 3e-4
BETAS = (0.9, 0.999)
EPS = 1e-8
WEIGHT_DECAY = 0.01

# GRADIENT_CLIPPING
L2_NORM = 1.0

DEVICE = torch.device("mps")

DATA_TYPE = torch.bfloat16

PROFILE = False
if PROFILE:
    logging.warning("Profiling is on")

torch.manual_seed(69)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

################ Settings Done #################

weights = {
    'token_embeddings.weight': torch.empty(VOCAB_SIZE, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
    'ln_final.weight': torch.ones(D_MODEL, device=DEVICE, dtype=DATA_TYPE),
    'lm_head.weight': torch.empty(VOCAB_SIZE, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0, std=0.02),
}
for i in range(NUM_LAYERS):
    layer_weights = {
        f'layers.{i}.attn.q_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.attn.k_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.attn.v_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.attn.output_proj.weight': torch.empty(D_MODEL, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ffn.w1.weight': torch.empty(D_FF, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ffn.w2.weight': torch.empty(D_MODEL, D_FF, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ffn.w3.weight': torch.empty(D_FF, D_MODEL, device=DEVICE, dtype=DATA_TYPE).normal_(mean=0.0, std=0.02),
        f'layers.{i}.ln1.weight': torch.ones(D_MODEL, device=DEVICE, dtype=DATA_TYPE),
        f'layers.{i}.ln2.weight': torch.ones(D_MODEL, device=DEVICE, dtype=DATA_TYPE),
    }
    weights |= layer_weights

logging.info("Init model...")
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

logging.info("Init optimizer...")
optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    betas=BETAS,
    weight_decay=WEIGHT_DECAY,
    eps=EPS,
    device=DEVICE
)

# for param in model.parameters():
#     print(param.shape)

def train(checkpoint_path: Path = None):
    logging.info("training start")
    data_path = Path("../ch2/tokenized_tiny_story/result.npy")
    data = np.load(data_path, mmap_mode="r")
    checkpoint_dir = Path("checkpoints")

    start = 0
    if checkpoint_path is not None:
        start = load_checkpoint(checkpoint_path, model, optimizer)

    def train_loop(iteration:int):
        batch, target = get_batch(data, batch_size=32, context_length=CONTEXT_LENGTH, device="mps")

        if PROFILE:
            torch.mps.synchronize()
            t_forward = time.perf_counter()
        result = model.forward(batch.long())
        if PROFILE:
            torch.mps.synchronize()
            logging.info(f"forward: {time.perf_counter() - t_forward:.4f}s")

        # entropy
        if PROFILE:
            torch.mps.synchronize()
            t_entropy = time.perf_counter()
        result = einops.rearrange(result, "batch context_length vocab_size -> (batch context_length) vocab_size")
        target1 = einops.rearrange(target, "batch context_length -> (batch context_length)")
        entropy: torch.Tensor = cross_entropy(result, target1)
        if PROFILE:
            torch.mps.synchronize()
            logging.info(f"cross entropy: {time.perf_counter() - t_entropy:.4f}s")

        if PROFILE:
            torch.mps.synchronize()
            t_backward = time.perf_counter()
        entropy.backward()
        if PROFILE:
            torch.mps.synchronize()
            logging.info(f"backward: {time.perf_counter() - t_backward:.4f}s")

        gradient_clipping(model.parameters(), L2_NORM)

        if PROFILE:
            torch.mps.synchronize()
            t_optimize = time.perf_counter()
        optimizer.step()
        if PROFILE:
            torch.mps.synchronize()
            logging.info(f"optimize: {time.perf_counter() - t_optimize:.4f}s")

        optimizer.zero_grad()

        if iteration % 500 == 0:
            logging.info("saving checkpoint...")
            ckpt_path = checkpoint_dir/f"{iteration}.ckpt"
            save_checkpoint(model, optimizer, iteration, ckpt_path)
            logging.info(f"checkpoint {ckpt_path.__str__()} saved")
        if iteration % 10 == 0:
            logging.info(f"iteration: {iteration:06d}  loss: {entropy.item()}")

    # warm up
    logging.info("warm up...")
    for iteration in range(start+1, start+3):
        train_loop(iteration)
    logging.info("warm up finished")

    # with torch.mps.profiler.profile(
    #         mode="interval,event",
    #         wait_until_completed=False,
    # ):
    logging.info("profile start")
    for iteration in range(start+3, TOTAL_STEPS):
        train_loop(iteration)

    torch.mps.synchronize()

    # save_checkpoint(model, optimizer, TOTAL_STEPS, checkpoint_dir/f"{TOTAL_STEPS}.ckpt")


def decode(output: Float[Tensor, "context_length vocab_size"], vocab: list[bytes]):
    useful = output[-1]
    probability = softmax(useful, -1)
    choice = torch.multinomial(probability, num_samples=1).item()
    return vocab[choice]


def infer():
    checkpoint_dir = Path("checkpoints")
    data_path = Path("../ch2/tokenized_tiny_story/result.npy")
    data = np.load(data_path, mmap_mode="r")
    ckpt_path = checkpoint_dir/"2000.ckpt"
    load_checkpoint(ckpt_path, model, optimizer)
    batch, _ = get_batch(data, batch_size=1, context_length=CONTEXT_LENGTH, device="mps")

    tokenizer_file_path = "/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.json"
    tokenizer = Tokenizer.from_json(tokenizer_file_path)
    seed_ids = batch[0].tolist()
    print(tokenizer.decode(seed_ids), end="", flush=True)

    print("开始推理！")

    model.eval()
    context = batch.long()
    with torch.no_grad():
        try:
            while True:
                output = model.forward(context)
                next_logits = output[0, -1]
                probability = softmax(next_logits, -1, temp=0.5)
                next_id = torch.multinomial(probability, num_samples=1)
                context = torch.cat([context, next_id.view(1, 1)], dim=1)[:, -CONTEXT_LENGTH:]
                print(tokenizer.decode([next_id.item()]), end="", flush=True)
        except KeyboardInterrupt:
            print()


# train(checkpoint_path=Path("checkpoints/1000.ckpt"))
# infer()
train()