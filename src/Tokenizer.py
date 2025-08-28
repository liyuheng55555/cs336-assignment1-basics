import json
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

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] = None,
    ):
        """
        :param vocab_filepath: json file, {"!": 0, "\"": 1, "#": 2, "$": 3}
        :param merges_filepath: Ġ t\nĠ a\nh e
        :param special_tokens:
        :return:
        """
        vocab: dict[int, bytes] = {}
        merge_rules: list[Connection] = []
        with open(vocab_filepath, "r", encoding="utf-8") as file:
            vocab = json.load(file)
        with open(merges_filepath, "r", encoding="utf-8") as file:
            line = file.readline()
            a, b = line.split(" ")
            merge_rules.append((a.encode("utf-8"), b.encode("utf-8")))
        return Tokenizer(vocab, merge_rules, special_tokens)

    def encode(self, text: str) -> list[int]:
        result: list[int] = []
        word_list: list[str] = chunk_split(
            text, special_tokens=self.special_tokens
        )
        for word in word_list:
            token_list: TokenList = bytes_to_bytes_list(word.encode("utf-8"))
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
        return result

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        for text in iterable:
            for code in self.encode(text):
                yield code

    def decode(self, ids: list[int]) -> str:
        result: str = ""
        for id in ids:
            result += self.vocab[id]
        return result
