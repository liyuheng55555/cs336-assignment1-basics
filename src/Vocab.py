

class Vocab:
    def __init__(self):
        self._vocab: set[bytes] = set()
        self._counter: int = 0   # 从 0 开始计数

    def put(self, token: bytes) -> int:
        key = self._counter
        self._vocab.add(token)
        self._counter += 1
        return key

    # def get(self, key: int) -> bytes | None:
    #     return self._vocab.get(key)

    def contains(self, key: str) -> bool:
        return str in self._vocab

    # def get_vocab(self) -> dict[int, bytes]:
    #     return self._vocab

    def split_word(self, word: bytes) -> list[bytes]:
        result: list[bytes] = []
        i = 0
        n = len(word)
        while i < n:
            # 从当前位置开始，找最长的匹配
            match = None
            for j in range(n, i, -1):  # 从长到短尝试
                piece = word[i:j]
                if piece in self._vocab:
                    match = piece
                    break
            if match is None:
                # 如果没有匹配到，就退化成单字节
                match = word[i:i+1]
            result.append(match)
            i += len(match)
        return result

    def build_dict(self):
        d = {}
        index = 0
        for b in self._vocab:
            d[index] = b
            index += 1
        return d

    def __len__(self) -> int:
        return len(self._vocab)

    def __repr__(self) -> str:
        return f"Vocab(size={len(self)}, next_id={self._counter})"

    def __str__(self) -> str:
        return str(self._vocab)
