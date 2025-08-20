import regex as re

from src.Vocab import Vocab
from src.utils import connection_to_str, max_connection

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""



def bpe_train(
        input_path: str,
        vocab_size: int,
        special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    vocab = Vocab()
    merges: list[tuple[bytes, bytes]] = []

    for i in range(0,256):
        vocab.put(bytes([i]))
    for token in special_tokens:
        vocab.put(token.encode("utf-8"))

    with open(input_path, "r", encoding="utf-8") as f:
        raw_content = f.read()
    contents = re.split(re.escape("|".join(special_tokens)), raw_content)
    print(len(contents))

    all_words : dict[bytes, int] = {} # "xxx" -> nums
    for content in contents:
        words = re.findall(PAT, content)
        for word in words:
            byte = word.encode("utf-8")
            if byte not in all_words:
                all_words[byte] = 1
            else:
                all_words[byte] += 1

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



    # print(tuple_to_str(best_connection), max_nums)

    while len(vocab) < vocab_size:
        # calculate connections
        connections = {}
        for word, nums in all_words.items():
            split_words = vocab.split_word(word)
            for i in range(1, len(split_words)):
                connection: tuple[bytes,bytes] = (split_words[i-1], split_words[i])
                if connection not in connections:
                    connections[connection] = nums
                else:
                    connections[connection] += nums

        # find best connection
        best_connections: list[tuple[bytes,bytes]] = []
        max_nums = 0
        for connection, nums in connections.items():
            if nums > max_nums:
                best_connections = [connection]
                max_nums = nums
            elif nums == max_nums:
                best_connections.append(connection)
        if len(best_connections) > 1 or max(best_connections) == (b'e', b'n'):
            print(best_connections)
        best_connection = max(best_connections)
        # print(connection_to_str(best_connection), max_nums)
        merges.append(best_connection)
        # update vocab
        vocab.put(best_connection[0] + best_connection[1])

    print("============\n\n")

    return vocab.build_dict(), merges




