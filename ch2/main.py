import logging
from codecs import getincrementaldecoder
from pathlib import Path
from time import perf_counter

import numpy as np

from ch2.Tokenizer import Tokenizer


def encode_file_streaming(
    tokenizer: Tokenizer,
    input_path: Path,
    output_dir: Path,
    read_chunk_bytes: int = 128 * 1024 * 1024,
    shard_token_threshold: int = 8 * 1024 * 1024,
) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)

    decoder = getincrementaldecoder("utf-8")()
    carry = ""
    buffer: list[int] = []
    token_count = 0
    shard_id = 0
    shard_paths: list[Path] = []

    def flush_buffer() -> None:
        nonlocal shard_id, token_count, buffer
        if not buffer:
            return
        path = output_dir / f"{shard_id:06d}.npy"
        shard_paths.append(path)
        np.save(path, np.array(buffer, dtype=np.uint16))
        token_count += len(buffer)
        shard_id += 1
        buffer = []

    with open(input_path, "rb") as f:
        while True:
            raw = f.read(read_chunk_bytes)
            if not raw:
                break

            text = carry + decoder.decode(raw, final=False)
            lines = text.splitlines(keepends=True)
            carry = ""
            if lines and not text.endswith(("\n", "\r")):
                carry = lines.pop()

            for line in lines:
                buffer.extend(tokenizer.encode(line))
                if len(buffer) >= shard_token_threshold:
                    flush_buffer()

        tail = carry + decoder.decode(b"", final=True)
        if tail:
            buffer.extend(tokenizer.encode(tail))
    flush_buffer()

    final = np.lib.format.open_memmap(
        output_dir / "result.npy",
        mode="w+",
        dtype=np.uint16,
        shape=(token_count,),
    )

    offset = 0
    for shard in shard_paths:
        x = np.load(shard, mmap_mode="r")
        n = len(x)
        final[offset : offset + n] = x
        offset += n
    final.flush()

    return token_count


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = Path("/data/cs336/data/owt_train.txt")
    tokenizer_file_path = Path("/data/cs336/data/owt_train.json")
    output_dir = Path("/data/cs336/data/owt_train_tokenized")

    start = perf_counter()
    tokenizer = Tokenizer.from_json(str(tokenizer_file_path))
    t1 = perf_counter()

    total_tokens = encode_file_streaming(tokenizer, input_path, output_dir)
    t2 = perf_counter()

    result = np.load(output_dir / "result.npy", mmap_mode="r")
    load_time = t1 - start
    encode_time = t2 - t1
    total_time = t2 - start

    print(f"tokens: {total_tokens}")
    print(f"result shape: {result.shape}, dtype: {result.dtype}")
    print("Timing (seconds):")
    print(f"  load tokenizer: {load_time:.3f}")
    print(f"  encode+write  : {encode_time:.3f}")
    print(f"  total         : {total_time:.3f}")
