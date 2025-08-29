

# 开个新坑：从零实现并加速 BPE 训练（含数据结构优化）

> 好几年没刷公开课了，这次CS336又让我有了刷课的动力！本节从零开始训练一个 BPE 训练器，并围绕「预分词」与「merge」两大阶段做了系统优化。
> 本文记录实现要点、性能瓶颈与数据结构取舍。
> 
> 先约定命名：`['a','p','ple']`这样的单词拆分结果称为`TokenList`，`('p','ple')`这样的相邻Token连接关系称为`Connection`

## TL;DR（结论速览）

* 训练流程可拆成相解耦的两块：
  * **预分词**，pretokenize
  * **merge**，按频次合并 Connection
* 预分词阶段的优化点主要为**多进程并行**、**内存控制**、**结果缓存**
* merge 阶段的核心在于:
  * 逻辑上用 `dict[Connection, (count, contributors)]` 做**Connection 统计缓存**，只增量更新受影响的 tokenlists，可将性能拉到可用。
  * 工程上选择更**高效维护“Connection→频次”的数据结构**，以便快速找到频次最高的Connection，这里我选择使用**有序计数桶（SortedDict）** 结构，维护「count→set(Connection)」，最大值查询 O(1)、更新 O(log n)，在 owt_train 上优于堆（避免堆的惰性失效膨胀）。
* 实测中，merge 阶段 **90% 时间花在修改数据结构**：删除受影响的 Connection 占 \~30%，新增 Connection 占 \~70%。前期规则耗时明显更长，原因是高频组合（如 `t`+`h`）拥有**极多贡献者**。

---

## 环境与数据集

* **硬件**：MacBook Air M3 / 24GB / 512GB
* **数据**：

    * TinyStoriesV2-GPT4-train.txt
    * owt-train.txt

### 端到端用时（样例）

| 数据集                      | 预分词（并发）      | merge   |
| ------------------------ | ------------ | ------- |
| TinyStoriesV2-GPT4-train | \~46s（单机）    | \~5s    |
| owt\_train               | \~300s（6 进程） | \~1100s |

> 注：预分词结果缓存开启后，第二次运行可以直接跳过预分词阶段。

---

## 总体流程

1. **预分词（pretokenization）**

    * 使用讲义给出的正则表达式分词。
    * 多进程并行（我电脑是 8 核，开 6 进程不至于让电脑变卡:）。
    * OWT 数据集需注意**内存控制**（在官方 `pretokenization_example.py` 略改分块逻辑即可）。
    * 把结果**持久化缓存**（python内置`pickle`库），重复实验可直接复用。

   预分词输出主数据结构：

   ```python
   all_token_list: list[tuple[TokenList, Num]]
   # 例如: TokenList = ['a','p','p','l','e'], Num = 出现次数
   ```

2. **merge（按频次合并）**

    * 循环：统计所有 Connection 的频次 → 取最大 → 记录 merge\_rule → 按语义对 tokenlists 执行“合并” → 直至词表上限。

---

## 正确性坑：别用“最长贪心匹配”替代“语义合并”

我最初在更新 `all_token_list` 时，尝试“对每个 TokenList 用词表做**贪心最长匹配**”，
这会在第一个测试的第 21 次合并开始与标准答案出现细微偏差。
**正确做法**是：严格按刚生成的 `merge_rule` 的“**合并语义**”逐步更新，而不是用词表做贪心匹配。

---

## 性能优化 1：Connection 统计缓存（增量更新）

**痛点**：naive 版本每轮都**全扫** `all_token_list` 统计 Connection → 代价过大。

**改进**：引入缓存，只更新“受上一轮 `merge_rule` 影响的 TokenList”。

```python
# Connection → (count, contributors)
# contributors:set[Index] 哪些 TokenList 贡献了这个 Connection 的计数
cache: dict[Connection, tuple[int, set[Index]]]
```

* 每次合并后，仅定位并更新这些 contributors 对应的 Connection 统计，避免全量扫描。
* 这一步能把课程提供的性能测试拉到「1.5s 内通过」。

**但问题还在**：每轮仍需在缓存中**找频次最大**的 Connection；数据集下 Connection 数量可达 10 万级，全量遍历仍吃性能。

---

## 性能优化 2：有序计数桶（SortedDict）

为**快速拿最大频次**且**支持更新**，首先考虑了使用“堆 + 惰性删除”。这虽可行，但频繁更新会使得失效元素过多，进而让堆的尺寸**膨胀**。

最终选择了**有序计数桶**（计数为 key，Connection 集合作为 value），最大值查询 O(1)，更新 O(log n)。

```python
from sortedcontainers import SortedDict

class BucketMaxSD:
    def __init__(self):
        self.counts: dict[Connection, int] = {}      # Connection → count
        self.buckets = SortedDict()                  # count → set[Connection]

    # 典型操作：
    # - 取最大：self.buckets.peekitem(-1)  # O(1)
    # - 更新：在旧 count 的 set 中删除，在新 count 的 set 中插入  # O(log n)
```

### 复杂度与实现细节

* **取最大**：O(1)（`SortedDict` 维护有序键，尾部即最大）
* **更新**：

    * `counts` 字典 O(1)
    * `buckets` 中一次删除 + 一次插入，总体 O(log n)
* `SortedDict`（`sortedcontainers`）底层不是红黑树，而是**分块有序列表**（list-of-lists），通过控制块大小与两次二分实现近似 O(log n) 写入与 O(1) 取 min/max。

> 实测在 OWT 上，这套“有序计数桶”优于“堆 + 惰性删除”。工业界常见做法仍是堆，但通常会**定期重建**或**只维护 top-k** 来控尺寸与内存；若充分打磨，堆也可能更快。

---

这里也和 gpt5 进一步讨论了使用“**双向链表计数桶**”的方案，但观察发现：
- 本任务的计数分布在高频尾端**非常稀疏**，例如某次找到的数量最多的 Connection 为 123456 次，在 123456 对应的桶中就只有这么一个 Connection。
- 这就使得本任务的**新建/删除桶**非常常见，几乎每次取出一个Connection都要做一次删除和一次新建，链表维护成本不低（但是链表可以实现对尾端批量删除，有兴趣的同学可以进一步探索
-`SortedDict` 的时间复杂度在这种分布下更稳


## 关键观察

* **时间分布**：merge 阶段约 **90% 时间**用于“修改缓存结构”：
    * **~25%**：删除受 `merge_rule` 影响的旧 Connection；
    * **~65%**：新增合并后产生的新 Connection。
    * 其余步骤如根据 `merge_rule` 更新 `TokenList` 消耗时间很少，可能由于英文单词都较短，直接重新组建一个新的`TokenList`的开销也不大，如果单个词比较长就需要考虑用链表来实现原地更新了。
* **前期更慢**：早期规则（如 `t`+`h`）拥有**极多贡献者**，涉及的增量更新范围更大；随着训练推进，贡献者规模下降，单轮越来越快。

---

## 代码骨架

让gpt总结了一份代码骨架，供参考

```
function BPE_TRAIN(input_path, vocab_size, special_tokens) -> (id_to_token, merge_rules):
# 初始化
vocab ← empty Vocab()
merge_rules ← empty list

# 基础词表：单字节 & 特殊符号
for b in [0..255]:
    vocab.put(byte(b))
for tok in special_tokens:
    vocab.put(utf8(tok))

# 预分词（不关心缓存/IO细节）
all_bytes ← PRETOKENIZE(input_path, special_tokens)

# 增量统计所需的状态
last_contributors ← empty list        # 上一步合并对贡献的位置索引
pair_count_map ← empty PairCountDS    # 连接(二元 token 对) → 频次
contributors_map ← empty dict         # 连接 → {Index}

while SIZE(vocab) < vocab_size:
    # 用增量方式更新二元对的计数与贡献者集合
    UPDATE_COUNTS(
        all_bytes,
        last_contributors,
        pair_count_map,
        contributors_map,
        last_merge = LAST(merge_rules) or None
    )

    # 选择出现次数最多的连接（最佳合并）
    (best_connection, contributors) ← FIND_BEST(pair_count_map, contributors_map)

    # 记录此次合并对及其贡献者位置，供下轮增量更新使用
    last_contributors ← LIST(contributors)
    APPEND(merge_rules, best_connection)

    # 将合并后的新 token 加入词表
    new_token ← CONCAT(best_connection.left, best_connection.right)
    vocab.put(new_token)

    # （可选）进度汇报：省略细节

return vocab.build_dict(), merge_rules

```


## 一些观察、思考、延伸

* 能否进一步并行化？预分词已并行；merge 若做**并行合并**会改变序列与统计，训练结果与单线程版本不同所以可能无法通过课程测试。
* owt数据集会明显观察到早期合并更慢，这是因为高频 Connection 的**贡献者集合巨大**，导致增量更新涉及的 Connection 多、集合操作重。
* `sortedcontainers.SortedDict` 的分块有序实现特性，适合“取极值 + 频繁插入删除”的场景。
* 工业界训练 BPE 常用堆配合惰性失效、定期重建或 top-k 候选池；也有并行 merge，或者使用 **Unigram** 模型的实践。

---

## 结语

目前这版实现，在课程要求下我已比较满意：

* 预分词靠并发与缓存省时。
* merge 用“Connection 统计缓存 + 有序计数桶”稳住了最大值选取与更新成本。

如果你也在实现 BPE 训练器，希望这套“**先确保正确、后增量优化**”的思路能帮你少踩坑～
