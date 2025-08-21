import logging

import regex as re

from src.Vocab import Vocab
from src.type_define import Index, Connection, Num
from src.utils import connection_to_str, max_connection, word_to_bytes_list, merge_once, merge_by_one_rule, \
    bytes_list_to_connections

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_best_connection(
        connections_num_map: dict[Connection,Num],
        connections_contrib_map: dict[Connection,set[Index]]
) -> tuple[Connection, set[Index]]:
    best_connections: list[tuple[Connection, set[Index]]] = []
    max_nums = 0
    for connection, nums in connections_num_map.items():
        contributors_index: set[Index] = connections_contrib_map[connection]
        if nums > max_nums:
            best_connections = [(connection, contributors_index)]
            max_nums = nums
        elif nums == max_nums:
            best_connections.append((connection, contributors_index))
    return max(best_connections)

def update(
        all_bytes: list[tuple[list[bytes], int]],
        last_contributors_index: list[Index],
        connections_num_map: dict[Connection,Num],
    connections_contrib_map: dict[Connection,set[Index]],
        merge_rule: Connection
):
    if last_contributors_index == []:
        last_contributors_index = list(range(len(all_bytes)))
    for i in last_contributors_index:
        bytes_list: list[bytes] = all_bytes[i][0]
        nums: int = all_bytes[i][1]
        # 根据最新的一条合并规则进行合并
        new_bytes_list: list[bytes]
        if merge_rule is not None:
            new_bytes_list = merge_by_one_rule(bytes_list,merge_rule)
        else:
            new_bytes_list = bytes_list
        if new_bytes_list is None:
            new_bytes_list = bytes_list
        # 旧的connection需要从统计中清除
        if merge_rule is not None:
            old_connections = bytes_list_to_connections(bytes_list)
            for connection in old_connections:
                if connection in connections_num_map:
                    connections_num_map[connection] -= nums
                    if i in connections_contrib_map[connection]:
                        connections_contrib_map[connection].remove(i)
        # 然后新的connection加入统计
        new_connections = bytes_list_to_connections(new_bytes_list)
        for connection in new_connections:
            if connection not in connections_num_map:
                connections_num_map[connection] = 0
                connections_contrib_map[connection] = set()
            connections_num_map[connection] += nums
            connections_contrib_map[connection].add(i)
        all_bytes[i] = (new_bytes_list, nums)


def pre_tokenize(input_path: str, special_tokens: list[str]) -> list[tuple[list[bytes], int]]:
    with open(input_path, "rb") as f:
        raw_content = f.read().decode("utf-8", errors="ignore")

    pattern = "|".join(re.escape(token) for token in special_tokens)
    contents = [c for c in re.split(pattern, raw_content) if c]

    logging.info(f"特殊符号切分完成, 共{len(contents)}段", )

    all_words : dict[bytes, int] = {} # "xxx" -> nums
    for content in contents:
        for match in re.finditer(PAT, content):
            word = match.group()
            byte = word.encode("utf-8")
            if byte not in all_words:
                all_words[byte] = 1
            else:
                all_words[byte] += 1

    logging.info(f"正则解析完成")

    all_bytes: list[tuple[list[bytes], int]] = []
    for word, nums in all_words.items():
        all_bytes.append((word_to_bytes_list(word), nums))

    logging.info(f"预分词完成")

    return all_bytes


def bpe_train(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[Connection]]:
    vocab = Vocab()
    merge_rules: list[Connection] = []

    for i in range(0,256):
        vocab.put(bytes([i]))
    for token in special_tokens:
        vocab.put(token.encode("utf-8"))

    all_bytes = pre_tokenize(input_path, special_tokens)

    idx = 0
    last_contributors_index: list[Index] = []
    connections_num_map: dict[Connection,Num] = {}
    connections_contrib_map: dict[Connection,set[Index]] = {}
    while len(vocab) < vocab_size:
        idx += 1
        # calculate connections
        update(
            all_bytes,
            last_contributors_index,
            connections_num_map,
            connections_contrib_map,
            merge_rules[-1] if len(merge_rules) > 0 else None
        )

        # find best connection
        best_connection, contributors = find_best_connection(connections_num_map, connections_contrib_map)
        last_contributors_index = list(contributors)
        merge_rules.append(best_connection)

        # update vocab
        vocab.put(best_connection[0] + best_connection[1])

        print(idx)

    print("============\n\n")

    return vocab.build_dict(), merge_rules

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/bpe_example.txt", 270, ["[<|endoftext|>"])
_, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 512, ["[<|endoftext|>"])
# _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.txt", 512, ["[<|endoftext|>"])
print(merge)


