import logging
import time
from collections import deque
from multiprocessing import Process, Queue
from time import perf_counter

import regex as re

from cs336_basics.pretokenization_example import find_chunk_boundaries
from src.BucketMaxSD import BucketMaxSD
from src.Vocab import Vocab
from src.type_define import Index, Connection, Num, TokenList, GB
from src.utils import (
    bytes_to_bytes_list,
    merge_by_one_rule,
    bytes_list_to_connections,
)

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_best_connection(
    connections_num_map: BucketMaxSD,
    connections_contrib_map: dict[Connection, set[Index]],
) -> tuple[Connection, set[Index]]:
    logging.info(f"connections_num_map size {len(connections_num_map)}")
    max_connection, _ = connections_num_map.max_item()
    contributors_index = connections_contrib_map[max_connection]
    return max_connection, contributors_index


def update(
    all_bytes: list[tuple[TokenList, int]],
    last_contributors_index: list[Index],
    connections_num_map: BucketMaxSD,
    connections_contrib_map: dict[Connection, set[Index]],
    merge_rule: Connection,
):
    if not last_contributors_index:
        last_contributors_index = list(range(len(all_bytes)))
    for i in last_contributors_index:
        token_list: TokenList = all_bytes[i][0]
        nums: int = all_bytes[i][1]
        # 根据最新的一条合并规则进行合并
        new_bytes_list: TokenList
        if merge_rule is not None:
            new_bytes_list = merge_by_one_rule(token_list, merge_rule)
        else:
            new_bytes_list = token_list
        if new_bytes_list is None:
            new_bytes_list = token_list
        # 旧的connection从统计中清除
        if merge_rule is not None:
            old_connections = bytes_list_to_connections(token_list)
            for connection in old_connections:
                if connection in connections_num_map:
                    connections_num_map.decr(connection, nums)
                    if (
                        connection in connections_contrib_map
                        and i in connections_contrib_map[connection]
                    ):
                        connections_contrib_map[connection].remove(i)
                        if len(connections_contrib_map[connection]) == 0:
                            connections_contrib_map.pop(connection)
        # 新的connection加入统计
        new_connections = bytes_list_to_connections(new_bytes_list)
        for connection in new_connections:
            connections_num_map.incr(connection, nums)
            if connection not in connections_contrib_map:
                connections_contrib_map[connection] = set()
            connections_contrib_map[connection].add(i)
        all_bytes[i] = (new_bytes_list, nums)


def pre_tokenize_worker(
    queue: Queue,
    input_path: str,
    start: int,
    end: int,
    special_tokens: list[str],
):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")

    logging.info(f"start {start} to {end}")

    pattern = "|".join(re.escape(token) for token in special_tokens)
    contents = [c for c in re.split(pattern, chunk) if c]

    print(
        f"特殊符号切分完成, 共{len(contents)}段",
    )

    all_words: dict[bytes, int] = {}  # "xxx" -> nums
    for content in contents:
        for match in re.finditer(PAT, content):
            word = match.group()
            byte = word.encode("utf-8")
            all_words[byte] = all_words.get(byte, 0) + 1

    print(f"正则解析完成")

    queue.put(all_words)


def pre_tokenize(
    input_path: str, special_tokens: list[str], concurrency: int = 6
) -> list[tuple[TokenList, Num]]:
    queue: Queue = Queue()
    workers: deque[Process] = deque()
    with open(input_path, "rb") as f:
        boundaries = find_chunk_boundaries(
            f,
            concurrency,
            list(token.encode("utf-8") for token in special_tokens),
            max_memory_in_bytes=16 * GB,
        )
        logging.info(f"文件切分为{len(boundaries)-1}块")
        for start, end in zip(boundaries[:-1], boundaries[1:]):
            process = Process(
                target=pre_tokenize_worker,
                args=(queue, input_path, start, end, special_tokens),
            )
            workers.append(process)

    logging.info("子进程准备完毕")

    finished_worker_count = 0
    worker_count = len(workers)

    for _ in range(min(concurrency, worker_count)):
        workers.pop().start()
        time.sleep(0.5)

    logging.info(f"首批{concurrency}个进程启动")

    merged_result: dict[bytes, Num] = {}

    while finished_worker_count < worker_count:
        # 每结束一个进程，就开一个新进程
        d: dict[bytes, int] = queue.get()
        finished_worker_count += 1
        logging.info(f"第 {finished_worker_count} 个任务完成")
        if len(workers) != 0:
            workers.pop().start()
        for k, v in d.items():
            merged_result[k] = merged_result.get(k, 0) + v

    logging.info("结果汇总完毕")

    return list((bytes_to_bytes_list(k), v) for k, v in merged_result.items())


def bpe_train(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Connection]]:
    vocab = Vocab()
    merge_rules: list[Connection] = []

    for i in range(0, 256):
        vocab.put(bytes([i]))
    for token in special_tokens:
        vocab.put(token.encode("utf-8"))

    pre_tokenize_start = perf_counter()
    all_bytes = pre_tokenize(input_path, special_tokens)
    pre_tokenize_end = perf_counter()

    idx = 0
    last_contributors_index: list[Index] = []
    connections_num_map = BucketMaxSD()
    connections_contrib_map: dict[Connection, set[Index]] = {}
    while len(vocab) < vocab_size:
        idx += 1
        # calculate connections
        update(
            all_bytes,
            last_contributors_index,
            connections_num_map,
            connections_contrib_map,
            merge_rules[-1] if len(merge_rules) > 0 else None,
        )

        # find best connection
        best_connection, contributors = find_best_connection(
            connections_num_map, connections_contrib_map
        )
        last_contributors_index = list(contributors)
        merge_rules.append(best_connection)

        # update vocab
        vocab.put(best_connection[0] + best_connection[1])

        # print(idx)

    calculate_end = perf_counter()

    print(f"预分词耗时: {pre_tokenize_end - pre_tokenize_start:.6f} 秒")
    print(f"计算耗时：{calculate_end - pre_tokenize_end:.6f} 秒")

    return vocab.build_dict(), merge_rules


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    main_start = perf_counter()
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/bpe_example.txt", 270, ["<|endoftext|>"])
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 512, ["<|endoftext|>"])
    _, merge = bpe_train(
        "/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt",
        10000,
        ["<|endoftext|>"],
    )
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/owt_valid.txt", 10000, ["<|endoftext|>"])
    # _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/owt_train.txt", 10000, ["<|endoftext|>"])
    # print(merge)
    main_end = perf_counter()
    print(f"总耗时: {main_end - main_start:.6f} 秒")
