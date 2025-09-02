import logging
from time import perf_counter

from cs336_basics.pretokenization_example import find_chunk_boundaries
from ch2.Tokenizer import Tokenizer
from ch2.type_define import GB

if __name__ == '__main__':

    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )

    input_path = "/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-valid.txt"
    # input_path = "/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/owt_valid.txt"
    # tokenizer_file_path = "/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/TinyStoriesV2-GPT4-train.json"
    tokenizer_file_path = "/Users/liyuheng/Documents/cs336/cs336-assignment1-basics/data/owt_train.json"
    with open(input_path, "rb") as f:
        content: str = f.read().decode("utf-8")

    start = perf_counter()

    tokenizer = Tokenizer.from_json(tokenizer_file_path)

    t1 = perf_counter()

    # ss = "Once upon a time there was a little boy named Ben. Ben loved to explore the world around him. He saw many amazing things, like beautiful vases that were on display in a store. One day, Ben was walking through the store when he came across a very special vase. When Ben saw it he was amazed!He said, “Wow, that is a really amazing vase! Can I buy it?”The shopkeeper smiled and said, “Of course you can. You can take it home and show all your friends how amazing it is!” So Ben took the vase home and he was so proud of it! He called his friends over and showed them the amazing vase. All his friends thought the vase was beautiful and couldn't believe how lucky Ben was. And that's how Ben found an amazing vase in the store!         <|endoftext|>           Once upon a time, there was a reliable otter named Ollie. He lived in a river with his family. They all loved to play and swim together. One day, Ollie's mom said, \"Ollie, hurry and get some fish for dinner!\" Ollie swam fast to catch fish. He saw his friend, the duck. \"Hi, Ollie!\" said the duck. \"Hi, duck!\" said Ollie. \"I need to hurry and catch fish for my family.\" While Ollie was catching fish, he found a big shiny stone. He thought, \"This is not a fish, but it is so pretty!\" Ollie took the shiny stone home to show his family. They all looked at the shiny stone and smiled. The shiny stone made everyone happy, and they forgot about the fish for dinner."
    e: list[int] = tokenizer.encode(content)

    t2 = perf_counter()

    print(len(e))
    print(len(content) / len(e))

    s = tokenizer.decode(e)

    t3 = perf_counter()

    # 打印阶段耗时与总耗时
    load_time = t1 - start
    encode_time = t2 - t1
    decode_time = t3 - t2
    total_time = t3 - start
    print("Timing (seconds):")
    print(f"  load tokenizer: {load_time:.3f}")
    print(f"  encode        : {encode_time:.3f}")
    print(f"  decode        : {decode_time:.3f}")
    print(f"  total         : {total_time:.3f}")

    assert content == s

