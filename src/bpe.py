import regex as re

from src.Vocab import Vocab
from src.type_define import Index, Connection, Num
from src.utils import connection_to_str, max_connection, word_to_bytes_list, merge_once, merge_by_one_rule, \
    bytes_list_to_connections

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


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


    with open(input_path, "rb") as f:
        raw_content = f.read().decode("utf-8", errors="ignore")


    pattern = "|".join(re.escape(token) for token in special_tokens)
    contents = [c for c in re.split(pattern, raw_content) if c]


    all_words : dict[bytes, int] = {} # "xxx" -> nums
    for content in contents:
        words = re.findall(PAT, content)
        for word in words:
            byte = word.encode("utf-8")
            if byte not in all_words:
                all_words[byte] = 1
            else:
                all_words[byte] += 1

    # print(tuple_to_str(best_connection), max_nums)

    idx = 0

    all_bytes: list[tuple[list[bytes], int]] = []
    for word, nums in all_words.items():
        all_bytes.append((word_to_bytes_list(word), nums))

    last_contributors_index: list[Index] = []
    connections_num_map: dict[Connection,Num] = {}
    connections_contrib_map: dict[Connection,set[Index]] = {}
    while len(vocab) < vocab_size:
        idx += 1
        # calculate connections
        if last_contributors_index == []:
            last_contributors_index = list(range(len(all_bytes)))
        for i in last_contributors_index:
            bytes_list: list[bytes] = all_bytes[i][0]
            nums: int = all_bytes[i][1]
            # 根据最新的一条合并规则进行合并
            new_bytes_list: list[bytes]
            if len(merge_rules) > 0:
                new_bytes_list = merge_by_one_rule(bytes_list, merge_rules[-1])
            else:
                new_bytes_list = bytes_list
            if new_bytes_list is None:
                new_bytes_list = bytes_list
            # 旧的connection需要从统计中清除
            if len(merge_rules) > 0:
                old_connections = bytes_list_to_connections(bytes_list)
                for connection in old_connections:
                    if connection in connections_num_map:
                        connections_num_map[connection] -= nums
                        # if i not in connections_contrib_map[connection]:
                        #     print("????")
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
        # if idx == 21:
        #     # 计算完成 connections 后，先看三对的计数
        #     def g(p): return connections.get(p, 0)
        #     print("21th-candidates:", {b'(o,u)': g((b'o',b'u')), b'(i,t)': g((b'i',b't')), b'(e,n)': g((b'e',b'n'))})
        #
        #     # 同时把贡献最大的前若干预分词找出来，看看是谁在“抬” (e,n)/(o,u)
        #     from collections import Counter
        #     contributors: dict[tuple[bytes, bytes], list[tuple[int,bytes]]] = {}
        #     for word, nums in all_words.items():
        #         toks = vocab.split_word(word)
        #         for i in range(1, len(toks)):
        #             pair = (toks[i-1], toks[i])
        #             if pair in {(b'o',b'u'), (b'i',b't'), (b'e',b'n')}:
        #                 if pair not in contributors:
        #                     contributors[pair] = []
        #                 contributors[pair].append((nums, word))
        #     for key, value in contributors.items():
        #         print(key)
        #         print(sorted(value, key=lambda x: x[0], reverse=True))
        #         print(sum(x[0] for x in value))

        # find best connection
        best_connections: list[tuple[Connection, set[Index]]] = []
        max_nums = 0
        for connection, nums in connections_num_map.items():
            contributors_index: set[Index] = connections_contrib_map[connection]
            if nums > max_nums:
                best_connections = [(connection, contributors_index)]
                max_nums = nums
            elif nums == max_nums:
                best_connections.append((connection, contributors_index))
        # if len(best_connections) > 1 or max(best_connections) == (b'e', b'n'):
        #     print(best_connections, max_nums)
        best_connection = max(best_connections)
        last_contributors_index = list(best_connection[1])
        # connections_num_map.pop(best_connection[0])
        # connections_contrib_map.pop(best_connection[0])
        # print(connection_to_str(best_connection), max_nums)
        print(idx)
        merge_rules.append(best_connection[0])
        # update vocab
        vocab.put(best_connection[0][0] + best_connection[0][1])

    print("============\n\n")

    return vocab.build_dict(), merge_rules


# _, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/bpe_example.txt", 270, ["[<|endoftext|>"])
_, merge = bpe_train("/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt", 512, ["[<|endoftext|>"])
print(merge)

# sorted_words = sorted(all_words.items(), key=lambda x: x[1], reverse=True)
# print(sorted_words[:10])

# initial_connections : dict[tuple[bytes,bytes], int] = {} # [a,b] -> nums
# for word, nums in all_words.items():
#     for i in range(1, len(word)):
#         connection: tuple[bytes,bytes] = (word[i-1:i], word[i:i+1])
#         if connection not in initial_connections:
#             initial_connections[connection] = nums
#         else:
#             initial_connections[connection] += nums

# sorted_connections = sorted(connections.items(), key=lambda x: x[1], reverse=True)
# for i in range(20):
#     t = sorted_connections[i][0]
#     n = sorted_connections[i][1]
#     print(tuple_to_str(t), n)

