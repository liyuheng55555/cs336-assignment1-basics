import json
import codecs
from collections import deque
from multiprocessing import Process
from multiprocessing import Queue
from typing import Iterator, Iterable

from src.type_define import TokenList, Connection
from src.utils import (
    chunk_split,
    bytes_to_bytes_list,
    bytes_list_to_connections,
    merge_by_one_rule,
)


class Tokenizer:

    def __init__(
        self,
        vocab: dict[int, bytes],
        merge_rules: list[Connection],
        special_tokens=None,
    ):
        self.vocab: dict[int, bytes] = vocab
        self.reverse_vocab: dict[bytes, int] = {v: k for k, v in vocab.items()}
        self.merge_rules: list[Connection] = merge_rules
        self.merge_rules_index: dict[Connection, int] = {
            t: i for i, t in enumerate(merge_rules)
        }
        self.MERGE_RULE_NOT_FOUND = 9999999
        self.special_tokens: list[str] = special_tokens
        self.special_tokens_set: set[str] = (
            set(special_tokens) if special_tokens is not None else set()
        )

    @classmethod
    def from_json(cls, json_filepath: str):
        """从合并的 JSON 文件构建 Tokenizer。

        该 JSON 文件由 bpe_train.py 写出，包含 vocab（bytes 以整型数组表示）、
        merges（两端 bytes 同样为整型数组）以及可选的 special_tokens。
        """
        from src.bpe_train import load_bpe_json

        vocab, merges, special_tokens = load_bpe_json(json_filepath)
        return cls(vocab, merges, special_tokens)

    def encode_worker(self, queue: Queue, index: int, words: Iterable[str]):
        result: list[int] = []
        for word in words:
            if word not in self.special_tokens_set:
                token_list: TokenList = bytes_to_bytes_list(
                    word.encode("utf-8")
                )
                while True:
                    connections: list[Connection] = bytes_list_to_connections(
                        token_list
                    )
                    chosen_merge_rule = None
                    min_index = self.MERGE_RULE_NOT_FOUND
                    for connection in connections:
                        if connection in self.merge_rules_index:
                            if self.merge_rules_index[connection] < min_index:
                                min_index = self.merge_rules_index[connection]
                                chosen_merge_rule = connection
                    if min_index != self.MERGE_RULE_NOT_FOUND:
                        token_list = merge_by_one_rule(
                            token_list, chosen_merge_rule
                        )
                    else:
                        break
                for token in token_list:
                    result.append(self.reverse_vocab[token])
            else:
                result.append(self.reverse_vocab[word.encode("utf-8")])
        queue.put((index, result))

    def encode(self, text: str) -> list[int]:

        word_list: list[str] = chunk_split(
            text, special_tokens=self.special_tokens
        )
        THRESHOLD = 10000

        split_word_list = [word_list[i:i+THRESHOLD] for i in range(0, len(word_list), THRESHOLD)]

        queue = Queue()
        workers: deque[Process] = deque()
        for i,words in enumerate(split_word_list):
            worker = Process(target=self.encode_worker, args=(queue, i, words))
            workers.append(worker)

        for _ in range(6):
            workers.pop().start()

        result_list: list[list[int]] = [[]] * len(workers)
        finished_worker_count = 0
        while finished_worker_count < len(workers):
            i, words = queue.get()
            result_list[i] = words
            if len(workers) != 0:
                workers.pop().start()

        result: list[int] = []
        for x in result_list:
            result.extend(x)

        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for code in self.encode(text):
                yield code

    def encode_file(self, input_path: str)  -> Iterator[int]:
        pass

    def decode(self, ids: Iterable[int]) -> str:
        buf = bytearray()
        for i in ids:
            buf.extend(self.vocab[i])
        return bytes(buf).decode("utf-8", errors="replace")

