# DataPelago Nucleus Coding 面试准备(Python 版)

> Every coding solution below is implemented and tested in Python.
> 代码兼容 Python 3.9+,每一段都是自包含的,末尾带 `assert`,可以直接复制运行。

## 0. 先读这一节

**这份材料的来源与可信度**

- 题目是我按"数据引擎 / 查询引擎公司的 coding 面试"推测出来的,**not published by DataPelago**。DataPelago 没有公开过这一轮的题目,我也不知道面试官会出什么。
- 我不知道这一轮的具体形式(白板、在线编辑器、共享 IDE)和语言。你选了 Python,这是合理的。
- Nucleus 内部用什么语言实现,我没有核实到公开来源。不要在面试里断言它是 Rust 或 C++,可以直接问。
- 标签含义:
  - `[General]`:行业通用知识,不依赖某个具体代码库。
  - `[Source]`:我读过本地代码(Gluten、Sail、Ballista)后确认的事实,会写出出处。
  - `[Ask]`:不确定,建议向面试官提问确认。

**怎么读**

1. 先读第 1 节(做题方法),它比任何一道题都重要。
2. 第 2 节 13 道编码题,每题:题目、常见追问、思路、Python 解法(带测试)、复杂度、和引擎的联系。
3. 第 3 节是口头概念题,面试官常在写完代码后追问。
4. 第 4 节 Python 的坑,第 5 节反问面试官的问题,第 6 节面试当天清单。

---

## 1. 做题方法(每道题都用)

1. **先问清楚(1–2 分钟):** 输入规模?数据是否有序?能否放进内存?null 怎么处理?重复 key 怎么算?输入是列表还是迭代器?
2. **先说思路和复杂度,再写代码。** 说出为什么选这个方案,以及你放弃的另一个方案。
3. **写完马上测试:** 空输入、单行、全部重复、全部 null、倾斜 key、超大输入。
4. **主动讲扩展:** "内存不够怎么办"、"怎么并行"、"怎么向量化"。数据引擎面试官最看重这个。
5. **边写边说。** 卡住时讲出你在权衡什么,不要沉默。
6. **接口先行:** 先写函数签名和一两个例子,再写实现,面试官更容易跟上。

数据引擎里反复出现的几个主题,几乎每道题都能往上靠:

| 主题 | 一句话 |
|---|---|
| 流式 / 迭代器 | 算子一批一批地拉(pull),不要一次物化全部数据 |
| 内存预算 | 任何会增长的结构(hash 表、排序缓冲)都要能 spill |
| 分区 | 按 key 切分后,每个分区可以独立处理、独立重试 |
| 倾斜 | 最大的分区决定总耗时,要能检测和拆分 |
| 向量化 | 一批一批处理,摊薄函数调用和分支开销 |

---

## 2. 编码题

### Q1. 实现 Hash Join(inner / left / semi / anti)

**题目:** 给两个"表"(每行是 `dict`),按某个列做 equi-join,支持 `inner`、`left`、`semi`、`anti` 四种类型。

**常见追问:**
- null key 怎么处理?(SQL 语义:null 不等于任何值,包括 null)
- 重复 key 会怎样?(产生笛卡尔积)
- 哪一侧做 build?(较小的一侧)
- 内存放不下怎么办?(分区后逐个分区 join,即 grace hash join)
- `NOT IN` 和 `NOT EXISTS` 的 anti join 语义有什么区别?(`NOT IN` 遇到 null 要返回空结果,这叫 null-aware anti join)

**思路:** 先扫 build 侧建 `key -> [rows]` 的哈希表,跳过 null key;再扫 probe 侧逐行查表。semi 只关心有没有匹配,anti 相反。

**Python Solution:**

```python
from collections import defaultdict


def hash_join(left, right, key, how="inner"):
    """Equi-join two iterables of dict on `key`. Build side is `right`."""
    if how not in ("inner", "left", "semi", "anti"):
        raise ValueError(f"unsupported join type: {how}")
    table = defaultdict(list)
    for r in right:
        k = r.get(key)
        if k is not None:  # SQL semantics: null never matches
            table[k].append(r)
    for l in left:
        k = l.get(key)
        matches = table.get(k) if k is not None else None
        if how == "inner":
            for r in matches or ():
                yield {**l, **r}
        elif how == "left":
            if matches:
                for r in matches:
                    yield {**l, **r}
            else:
                yield dict(l)
        elif how == "semi":
            if matches:
                yield l
        else:  # anti
            if not matches:
                yield l


def grace_hash_join(left, right, key, num_partitions, how="inner"):
    """Partition both sides by hash(key), then join partition by partition.

    Each partition pair fits in memory even if the whole input does not.
    """
    lparts = [[] for _ in range(num_partitions)]
    rparts = [[] for _ in range(num_partitions)]
    for l in left:
        k = l.get(key)
        lparts[0 if k is None else hash(k) % num_partitions].append(l)
    for r in right:
        k = r.get(key)
        rparts[0 if k is None else hash(k) % num_partitions].append(r)
    for lp, rp in zip(lparts, rparts):
        yield from hash_join(lp, rp, key, how)


users = [
    {"id": 1, "name": "a"},
    {"id": 2, "name": "b"},
    {"id": 2, "name": "b2"},  # duplicate key on the probe side
    {"id": None, "name": "n"},  # null key never matches
    {"id": 9, "name": "z"},  # no match on the build side
]
orders = [
    {"id": 1, "amt": 10},
    {"id": 2, "amt": 20},
    {"id": 2, "amt": 21},  # duplicate key on the build side
    {"id": None, "amt": 99},
]


def canon(rows):
    # repr() keeps the sort total even when values mix None and ints
    return sorted((tuple(sorted(r.items())) for r in rows), key=repr)


inner = list(hash_join(users, orders, "id", "inner"))
# id=1 -> 1 row, id=2 appears twice on both sides -> 2 x 2 = 4 rows
assert len(inner) == 5
assert all(r["id"] is not None for r in inner)

left = list(hash_join(users, orders, "id", "left"))
assert len(left) == 5 + 2  # unmatched: id=None and id=9
assert sum(1 for r in left if "amt" not in r) == 2

assert [r["name"] for r in hash_join(users, orders, "id", "semi")] == ["a", "b", "b2"]
assert [r["name"] for r in hash_join(users, orders, "id", "anti")] == ["n", "z"]

# grace hash join yields the same rows (order may differ)
for how in ("inner", "left", "semi", "anti"):
    assert canon(grace_hash_join(users, orders, "id", 3, how)) == canon(
        hash_join(users, orders, "id", how)
    )

assert list(hash_join([], orders, "id")) == []
assert list(hash_join(users, [], "id", "left")) == [dict(u) for u in users]
```

**复杂度:** 时间 O(|L| + |R| + 输出行数),空间 O(|R|)。
**和引擎的联系:** `[Source]` Gluten 的 Velox 后端把 Spark 的 shuffled hash join 和 broadcast hash join 都转成 Velox 的 `HashJoinNode`,支持 Inner、Left、Right、Full、LeftSemi、RightSemi、LeftAnti(含 null-aware anti)。广播 join 在 executor 上还会按 `hashtableid` 缓存预先构建的 hash 表,让多个 task 共享。

---

### Q2. 排序合并连接(Sort-Merge Join)与 k 路归并

**题目:** (a) 实现 `k_way_merge`,把 k 个已排序的迭代器合并成一个有序流。(b) 在两个已按 key 排序的列表上实现 sort-merge join,正确处理重复 key。

**常见追问:**
- k 路归并的复杂度?(O(N log k) 时间,O(k) 内存)
- 为什么堆里要放下标?(避免两个相等元素去比较迭代器本身,迭代器不可比较)
- sort-merge join 在什么情况下比 hash join 好?(两边都已排序,或者两边都很大、build 侧放不进内存)

**思路:** 归并用最小堆,每个迭代器各留一个"队首"元素。merge join 用双指针,遇到相等 key 时先把两边各自的这一组重复 key 找出来,再做笛卡尔积。

**Python Solution:**

```python
import heapq

_DONE = object()  # sentinel: safe even if the data itself contains None


def k_way_merge(iterables):
    heap = []
    for idx, iterable in enumerate(iterables):
        it = iter(iterable)
        first = next(it, _DONE)
        if first is not _DONE:
            # idx breaks ties so that iterators are never compared
            heap.append((first, idx, it))
    heapq.heapify(heap)
    while heap:
        value, idx, it = heapq.heappop(heap)
        yield value
        nxt = next(it, _DONE)
        if nxt is not _DONE:
            heapq.heappush(heap, (nxt, idx, it))


def sort_merge_join(left, right, key):
    """Inner join; both inputs must already be sorted by `key`."""
    out = []
    i = j = 0
    while i < len(left) and j < len(right):
        lk, rk = left[i][key], right[j][key]
        if lk < rk:
            i += 1
        elif lk > rk:
            j += 1
        else:
            i2 = i
            while i2 < len(left) and left[i2][key] == lk:
                i2 += 1
            j2 = j
            while j2 < len(right) and right[j2][key] == lk:
                j2 += 1
            for l in left[i:i2]:
                for r in right[j:j2]:
                    out.append({**l, **r})
            i, j = i2, j2
    return out


assert list(k_way_merge([[1, 4, 9], [2, 3, 10], [], [0, 5]])) == [0, 1, 2, 3, 4, 5, 9, 10]
assert list(k_way_merge([])) == []
assert list(k_way_merge([[], []])) == []
assert list(k_way_merge([[1, 1], [1]])) == [1, 1, 1]
assert list(k_way_merge([[None, None]])) == [None, None]  # sentinel handles None

L = [{"k": 1, "a": "x"}, {"k": 2, "a": "y"}, {"k": 2, "a": "z"}, {"k": 5, "a": "w"}]
R = [{"k": 2, "b": 1}, {"k": 2, "b": 2}, {"k": 3, "b": 3}, {"k": 5, "b": 4}]
joined = sort_merge_join(L, R, "k")
assert len(joined) == 2 * 2 + 1
assert {(r["a"], r["b"]) for r in joined if r["k"] == 2} == {
    ("y", 1), ("y", 2), ("z", 1), ("z", 2)
}
assert sort_merge_join([], R, "k") == []
```

**复杂度:** 归并 O(N log k);merge join 在输入已排序时 O(|L| + |R| + 输出)。
**和引擎的联系:** `[Source]` Gluten 在 Velox 后端里用 `isSMJ=` 标记把 sort-merge join 转成 `MergeJoinNode`。但默认配置 `forceShuffledHashJoin=true` 会把大多数 SMJ 改写成 SHJ,见第 3 节 C1。

---

### Q3. 外部排序(数据比内存大)

**题目:** 输入是一个很长的迭代器,内存里最多放 `chunk_size` 条,请输出全局有序的结果。

**常见追问:**
- 为什么是"分块排序 + k 路归并"?(每块在内存里排序,写成有序的 run,最后归并所有 run)
- 如果 run 的数量很多,文件句柄不够怎么办?(多趟归并:先把若干个 run 合并成更大的 run)
- 怎么让 run 更少?(用替换选择排序,run 平均长度约为内存的 2 倍)
- 怎么保证清理临时文件?(`try/finally`)

**Python Solution:**

```python
import heapq
import itertools
import os
import pickle
import random
import shutil
import tempfile


def _read_run(path):
    with open(path, "rb") as f:
        while True:
            try:
                yield pickle.load(f)
            except EOFError:
                return


def external_sort(iterable, chunk_size, key=None):
    """Sort an iterable whose size may exceed memory: sorted runs + k-way merge."""
    tmpdir = tempfile.mkdtemp(prefix="extsort_")
    try:
        runs = []
        it = iter(iterable)
        while True:
            chunk = list(itertools.islice(it, chunk_size))
            if not chunk:
                break
            chunk.sort(key=key)
            path = os.path.join(tmpdir, f"run{len(runs)}.pkl")
            with open(path, "wb") as f:
                for item in chunk:
                    pickle.dump(item, f)
            runs.append(path)
        # heapq.merge pulls lazily, so memory stays O(number of runs)
        yield from heapq.merge(*(_read_run(p) for p in runs), key=key)
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)


rng = random.Random(7)
data = [rng.randint(0, 10_000) for _ in range(5_000)]
assert list(external_sort(iter(data), chunk_size=300)) == sorted(data)
assert list(external_sort(iter(data), chunk_size=10**9)) == sorted(data)  # single run
assert list(external_sort(iter([]), chunk_size=10)) == []
assert list(external_sort(iter([3, 1, 2]), chunk_size=1)) == [1, 2, 3]

rows = [{"k": rng.randint(0, 50), "i": i} for i in range(1_000)]
out = list(external_sort(iter(rows), chunk_size=64, key=lambda r: r["k"]))
assert [r["k"] for r in out] == sorted(r["k"] for r in rows)
assert len(out) == len(rows)

# the temporary directory is removed once the generator is exhausted
before = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("extsort_")}
list(external_sort(iter(data), chunk_size=500))
after = {d for d in os.listdir(tempfile.gettempdir()) if d.startswith("extsort_")}
assert after <= before
```

**复杂度:** 时间 O(N log N),磁盘 IO 读写各 O(N);内存 O(chunk_size + k)。
**和引擎的联系:** `[Source]` Gluten 的 `VeloxSortShuffleWriter` 在内存里缓冲、超过阈值就 spill,最后合并,思路相同。Ballista 的 sort shuffle 也按这个思路(`sort_shuffle/` 里有 buffer、spill、index)。

---

### Q4. 流式 Top-K

**题目:** 数据是一个很长的流,返回得分最高的 K 个元素,内存只能放 K 个。

**常见追问:**
- 为什么用最小堆而不是最大堆?(堆顶是当前第 K 大的"门槛",新元素只需和它比较)
- 复杂度?(O(N log K))
- 并行怎么做?(每个分区各自算 Top-K,再合并这些局部结果取 Top-K)
- 频次 Top-K 怎么做?(`Counter` 或 Count-Min Sketch + 堆)

**Python Solution:**

```python
import heapq
import random
from collections import Counter


def top_k(stream, k, key=lambda x: x):
    """Return the k items with the largest key, largest first."""
    if k <= 0:
        return []
    heap = []  # min-heap of (key, seq, item); heap[0] is the k-th largest so far
    for seq, item in enumerate(stream):
        entry = (key(item), seq, item)  # seq keeps tuples comparable
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[0] > heap[0][0]:
            heapq.heapreplace(heap, entry)
    return [item for _, _, item in sorted(heap, reverse=True)]


def merge_top_k(partials, k, key=lambda x: x):
    """Combine per-partition results (each already a top-k list)."""
    return top_k((x for p in partials for x in p), k, key)


rng = random.Random(3)
nums = [rng.randint(0, 1_000_000) for _ in range(10_000)]
assert top_k(iter(nums), 5) == sorted(nums, reverse=True)[:5]
assert top_k(iter(nums), 0) == []
assert top_k(iter([3, 1]), 5) == [3, 1]  # fewer than k items
assert top_k(iter([]), 3) == []

# partitioned execution gives the same answer as a single pass
parts = [nums[i::4] for i in range(4)]
partials = [top_k(iter(p), 5) for p in parts]
assert merge_top_k(partials, 5) == top_k(iter(nums), 5)

words = ["a"] * 5 + ["b"] * 3 + ["c"] * 3 + ["d"]
freq = Counter(words)
best = top_k(freq.items(), 2, key=lambda kv: kv[1])
assert best[0] == ("a", 5)
assert best[1][1] == 3  # "b" and "c" tie at 3
```

**复杂度:** 时间 O(N log K),空间 O(K)。
**和引擎的联系:** 分布式引擎里 `ORDER BY ... LIMIT K` 就是这样:每个分区先局部 Top-K,再合并,避免全局排序。

---

### Q5. LRU 缓存(哈希表加双向链表,支持按"代价"淘汰)

**题目:** 实现 LRU 缓存,`get`/`put` 都是 O(1)。追问:容量按字节数而不是条目数计算。

**常见追问:**
- 为什么 `OrderedDict` 之外还要会手写?(面试官常常禁止直接用 `OrderedDict`)
- 线程安全怎么做?(一把锁保护整个结构;更高并发用分片锁)
- 单个条目比总容量还大怎么办?(不缓存,而不是把整个缓存清空)

**Python Solution:**

```python
class _Node:
    __slots__ = ("key", "value", "cost", "prev", "next")

    def __init__(self, key=None, value=None, cost=0):
        self.key, self.value, self.cost = key, value, cost
        self.prev = self.next = None


class LRUCache:
    """Hash map + doubly linked list. Capacity is measured in `cost` units."""

    def __init__(self, capacity, cost_fn=lambda v: 1):
        self.capacity = capacity
        self.cost_fn = cost_fn
        self.used = 0
        self.map = {}
        self.head, self.tail = _Node(), _Node()  # sentinels: head=MRU side
        self.head.next, self.tail.prev = self.tail, self.head

    def _unlink(self, n):
        n.prev.next, n.next.prev = n.next, n.prev

    def _push_front(self, n):
        n.prev, n.next = self.head, self.head.next
        self.head.next.prev = n
        self.head.next = n

    def get(self, key):
        n = self.map.get(key)
        if n is None:
            return None
        self._unlink(n)
        self._push_front(n)
        return n.value

    def put(self, key, value):
        cost = self.cost_fn(value)
        if key in self.map:  # replace: drop the old entry first
            old = self.map.pop(key)
            self._unlink(old)
            self.used -= old.cost
        if cost > self.capacity:  # too big to cache at all
            return
        n = _Node(key, value, cost)
        self.map[key] = n
        self._push_front(n)
        self.used += cost
        while self.used > self.capacity:
            victim = self.tail.prev
            self._unlink(victim)
            del self.map[victim.key]
            self.used -= victim.cost


c = LRUCache(2)
c.put("a", 1)
c.put("b", 2)
assert c.get("a") == 1  # "a" becomes most recently used
c.put("c", 3)  # evicts "b"
assert c.get("b") is None and c.get("a") == 1 and c.get("c") == 3
c.put("a", 10)  # update keeps a single entry
assert c.get("a") == 10 and len(c.map) == 2

sized = LRUCache(10, cost_fn=len)  # capacity in characters
sized.put("k1", "aaaa")
sized.put("k2", "bbbb")
sized.put("k3", "cc")
assert sized.used == 10
sized.put("k4", "dddd")  # needs 4 more -> evicts k1 (the LRU entry)
assert sized.get("k1") is None and sized.get("k2") == "bbbb"
sized.put("huge", "x" * 11)  # larger than capacity: not cached, nothing evicted
assert sized.get("huge") is None and sized.get("k2") == "bbbb"
assert sized.used <= sized.capacity
```

**复杂度:** `get`/`put` 均摊 O(1)。
**和引擎的联系:** 元数据缓存、文件句柄缓存、列式 buffer pool 都是这个结构。按字节计费对应"内存预算"的概念。

---

### Q6. 拉模式(Pull)的流式算子:Scan、Filter、Project、Limit

**题目:** 用生成器实现按"批"(batch)流动的算子 `scan → filter → project → limit`,并证明 `limit` 满足后会停止拉取上游。

**常见追问:**
- 为什么按批而不是按行?(摊薄函数调用和分支开销,便于向量化)
- 为什么 `limit` 要提前停止?(避免上游多读数据)
- Pull 模型和 Push 模型的区别?(Pull 由下游驱动,简单、天然背压;Push 由上游驱动,适合分支多的计划和流式)

**Python Solution:**

```python
def scan(rows, batch_size, counter=None):
    """Produce lists of rows; `counter` records how many rows were actually pulled."""
    batch = []
    for row in rows:
        if counter is not None:
            counter[0] += 1
        batch.append(row)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def filter_op(batches, predicate):
    for batch in batches:
        out = [r for r in batch if predicate(r)]
        if out:
            yield out


def project_op(batches, fn):
    for batch in batches:
        yield [fn(r) for r in batch]


def limit_op(batches, n):
    remaining = n
    for batch in batches:
        if remaining <= 0:
            return  # stop pulling: the upstream generator is simply not resumed
        if len(batch) > remaining:
            batch = batch[:remaining]
        yield batch
        remaining -= len(batch)
        if remaining <= 0:
            return


def collect(batches):
    return [r for b in batches for r in b]


pulled = [0]
plan = limit_op(
    project_op(
        filter_op(scan(range(1_000_000), 100, pulled), lambda x: x % 2 == 0),
        lambda x: x * 10,
    ),
    5,
)
assert collect(plan) == [0, 20, 40, 60, 80]
assert pulled[0] <= 200  # nowhere near 1,000,000: limit stopped the pipeline early

assert collect(limit_op(scan(range(3), 2), 10)) == [0, 1, 2]
assert collect(limit_op(scan(range(10), 3), 0)) == []
assert collect(filter_op(scan([], 4), lambda x: True)) == []
assert collect(project_op(scan(range(5), 2), lambda x: -x)) == [0, -1, -2, -3, -4]
```

**复杂度:** 每个算子 O(批大小) 额外内存,总内存与批数无关。
**和引擎的联系:** Velox 和 DataFusion 都是这种"按批拉取"的算子树。`[Source]` Sail 的任务输出流(`TaskStreamSink`)和 Ballista 的 `ExecutionPlan::execute` 返回的 `RecordBatchStream` 也是这个模型。

---

### Q7. Hash 分区、合并小分区、检测倾斜

**题目:** (a) 把行按 key 哈希分到 N 个分区。(b) 给定每个分区的字节数,把相邻的小分区合并成大约 `target` 大小的组。(c) 找出倾斜分区。

**常见追问:**
- 为什么只合并"相邻"分区?(保持分区号的范围语义,元数据只需记录区间)
- Python 内置 `hash()` 能用吗?(字符串哈希每次进程启动都会变,跨进程、跨机器必须用稳定哈希)
- 倾斜怎么判断?(大于"非空分区大小的中位数 × 因子",并且超过一个绝对下限)

**Python Solution:**

```python
import zlib


def stable_hash(value):
    """Process-independent hash. Python's built-in hash() is salted per process."""
    return zlib.crc32(str(value).encode("utf-8"))


def hash_partition(rows, key, num_partitions):
    parts = [[] for _ in range(num_partitions)]
    for row in rows:
        parts[stable_hash(row[key]) % num_partitions].append(row)
    return parts


def coalesce_partitions(sizes, target):
    """Greedily merge adjacent partitions up to ~target bytes. Returns [(start, end)]."""
    if not sizes:
        return []
    if target <= 0:
        return [(i, i + 1) for i in range(len(sizes))]
    ranges, start, current = [], 0, 0
    for i, size in enumerate(sizes):
        if current > 0 and current + size > target:
            ranges.append((start, i))
            start, current = i, size
        else:
            current += size
    ranges.append((start, len(sizes)))
    return ranges


def detect_skew(sizes, factor, min_bytes):
    non_zero = sorted(s for s in sizes if s > 0)
    if not non_zero:
        return []
    median = non_zero[len(non_zero) // 2]
    return [i for i, s in enumerate(sizes) if s >= min_bytes and s > median * factor]


rows = [{"k": i % 7, "v": i} for i in range(700)]
parts = hash_partition(rows, "k", 4)
assert sum(len(p) for p in parts) == 700
keys_by_part = [{r["k"] for r in p} for p in parts]
for a in range(4):
    for b in range(a + 1, 4):
        assert keys_by_part[a].isdisjoint(keys_by_part[b])

assert coalesce_partitions([], 100) == []
assert coalesce_partitions([10, 20, 30], 0) == [(0, 1), (1, 2), (2, 3)]
assert coalesce_partitions([10] * 5, 100) == [(0, 5)]
assert coalesce_partitions([20, 25, 30, 15, 45, 10], 50) == [(0, 2), (2, 4), (4, 5), (5, 6)]
assert coalesce_partitions([5, 5, 200, 10, 10], 50) == [(0, 2), (2, 3), (3, 5)]  # oversize stays alone

assert detect_skew([10, 12, 500, 11, 10], factor=5.0, min_bytes=100) == [2]
assert detect_skew([10, 12, 500, 11, 10], factor=5.0, min_bytes=1000) == []
assert detect_skew([0, 0], factor=5.0, min_bytes=1) == []
```

**复杂度:** 分区 O(N);合并与倾斜检测 O(P log P)(P 是分区数)。
**和引擎的联系:** `[Source]` 这个合并算法和 Sail 的 `coalesce_shuffle_partitions` 以及倾斜检测 `detect_skew_partitions`(中位数乘 `skew_factor`,且超过 `min_skew_threshold`)一致。这是你自己在 Sail 分支里实现过的逻辑,面试里可以直接讲。

---

### Q8. 带内存预算的 Hash 聚合(partial aggregation + spill)

**题目:** 实现 `GROUP BY key` 的 `sum` 和 `count`。内存里最多放 `max_groups` 个分组,超出就把当前哈希表按 key 排序后写到磁盘(spill),最后合并所有 spill 和内存里剩余的部分。

**常见追问:**
- 为什么 spill 时要排序?(这样最后可以用 k 路归并,按 key 相邻合并,内存只需 O(k))
- 为什么 `avg` 不能直接合并?(要保存 `(sum, count)`,最后再相除)
- 分布式下怎么做?(两阶段:每个分区先 partial aggregate,按 key shuffle,再 final aggregate)

**Python Solution:**

```python
import heapq
import os
import pickle
import random
import shutil
import tempfile
from collections import defaultdict


class SpillingAggregator:
    """GROUP BY key with sum/count; spills sorted runs when the table is full."""

    def __init__(self, max_groups):
        self.max_groups = max_groups
        self.table = {}  # key -> [sum, count]
        self.runs = []
        self.tmpdir = tempfile.mkdtemp(prefix="aggspill_")

    def add(self, key, value):
        if key not in self.table and len(self.table) >= self.max_groups:
            self._spill()
        acc = self.table.setdefault(key, [0, 0])
        acc[0] += value
        acc[1] += 1

    def _spill(self):
        path = os.path.join(self.tmpdir, f"run{len(self.runs)}.pkl")
        with open(path, "wb") as f:
            pickle.dump(sorted(self.table.items()), f)  # sorted by key
        self.runs.append(path)
        self.table = {}

    def _iter_run(self, path):
        with open(path, "rb") as f:
            yield from pickle.load(f)

    def result(self):
        try:
            sources = [self._iter_run(p) for p in self.runs]
            sources.append(iter(sorted(self.table.items())))
            out = {}
            current_key, s, c = None, 0, 0
            started = False
            for key, (ps, pc) in heapq.merge(*sources, key=lambda kv: kv[0]):
                if started and key == current_key:
                    s, c = s + ps, c + pc
                else:
                    if started:
                        out[current_key] = (s, c)
                    current_key, s, c, started = key, ps, pc, True
            if started:
                out[current_key] = (s, c)
            return out
        finally:
            shutil.rmtree(self.tmpdir, ignore_errors=True)


rng = random.Random(11)
rows = [(rng.randint(0, 199), rng.randint(1, 100)) for _ in range(20_000)]

expected = defaultdict(lambda: [0, 0])
for k, v in rows:
    expected[k][0] += v
    expected[k][1] += 1

agg = SpillingAggregator(max_groups=25)  # 200 distinct keys -> many spills
for k, v in rows:
    agg.add(k, v)
assert len(agg.runs) > 1
got = agg.result()
assert got == {k: tuple(v) for k, v in expected.items()}

# no spill needed when the budget is large enough
big = SpillingAggregator(max_groups=10_000)
for k, v in rows:
    big.add(k, v)
assert big.runs == []
assert big.result() == got

empty = SpillingAggregator(max_groups=1)
assert empty.result() == {}
```

**复杂度:** 时间 O(N + S log S)(S 是 spill 的总条目数),内存 O(max_groups + run 数)。
**和引擎的联系:** `[Source]` Gluten 在 Velox 里通过内存仲裁触发 hash aggregation 的 spill,选择最大的消费者优先 spill,见第 3 节 C4。`[General]` Spark 的 `HashAggregate` 同样分 partial 和 final 两阶段。

---

### Q9. 带 spill 的内存池(Memory Pool with Arbitration)

**题目:** 实现一个内存池:多个消费者(算子)向它申请内存;池满时,让其他消费者 spill 腾出空间,**最大的消费者优先**;腾不出来就抛 OOM。

**常见追问:**
- 为什么选最大的?(一次 spill 释放最多,spill 次数最少)
- 申请者自己要不要 spill?(先让别人 spill,自己是最后手段)
- 并发申请怎么保证正确?(池级别的锁;更细要分层)
- 怎么避免 spill 期间死锁?(spill 回调不能再向池申请内存)

**Python Solution:**

```python
class MemoryPool:
    def __init__(self, capacity):
        self.capacity = capacity
        self.used = 0
        self.consumers = []

    def register(self, name, spill_fn=None):
        c = Consumer(name, self, spill_fn)
        self.consumers.append(c)
        return c

    def acquire(self, consumer, size):
        if size > self.capacity:
            raise MemoryError(f"{consumer.name}: request {size} exceeds pool {self.capacity}")
        while self.used + size > self.capacity:
            others = [c for c in self.consumers if c is not consumer and c.used > 0 and c.spill_fn]
            others.sort(key=lambda c: c.used, reverse=True)  # largest first
            if others:
                victim = others[0]
            elif consumer.used > 0 and consumer.spill_fn:
                victim = consumer  # last resort: spill the requester itself
            else:
                raise MemoryError(f"{consumer.name}: pool exhausted, nothing left to spill")
            if victim.spill() <= 0:
                raise MemoryError(f"{victim.name}: spill freed nothing")
        self.used += size
        consumer.used += size


class Consumer:
    def __init__(self, name, pool, spill_fn):
        self.name, self.pool, self.spill_fn, self.used = name, pool, spill_fn, 0
        self.spilled = 0

    def acquire(self, size):
        self.pool.acquire(self, size)

    def release(self, size):
        self.used -= size
        self.pool.used -= size

    def spill(self):
        freed = self.spill_fn(self) if self.spill_fn else 0
        if freed > 0:
            self.release(freed)
            self.spilled += freed
        return freed


def spill_all(consumer):
    return consumer.used  # pretend we wrote everything to disk


pool = MemoryPool(100)
sort = pool.register("sort", spill_all)
agg = pool.register("agg", spill_all)
pinned = pool.register("pinned")  # cannot spill

sort.acquire(60)
agg.acquire(30)
assert pool.used == 90

agg.acquire(30)  # needs 20 more -> the largest *other* consumer (sort) spills
assert sort.used == 0 and sort.spilled == 60
assert pool.used == 60

pinned.acquire(40)
assert pool.used == 100

try:
    pinned.acquire(1)  # the only spillable consumer is agg; it spills, then it fits
    assert agg.used == 0 and pool.used == 41
except MemoryError:
    raise AssertionError("agg should have been spilled to satisfy the request")

try:
    pool.acquire(pinned, 1000)
    raise AssertionError("a request larger than the pool must fail")
except MemoryError:
    pass

stuck = MemoryPool(10)
a = stuck.register("a")  # no spill callback
a.acquire(10)
try:
    a.acquire(1)
    raise AssertionError("expected OOM when nothing can be spilled")
except MemoryError:
    pass
assert stuck.used == 10
```

**复杂度:** 每次仲裁 O(C log C)(C 是消费者数)。
**和引擎的联系:** `[Source]` 我之前读 Gluten 源码写的指南(`datapelago_gluten_shuffle_memory_guide.md`)里记录了它的内存栈:`TreeMemoryConsumer` 和 `TreeMemoryTargets.spillTree`(先 SHRINK 再 SPILL,最大者优先)、`ThrowOnOomMemoryTarget`(OOM 前先 GC 并重试),以及 Velox 侧的 `ListenableArbitrator`。这道题就是它们的简化版。

---

### Q10. 把 (分区号, 行号) 打包进一个整数,并用计数排序按分区重排

**题目:** (a) 把 `partition_id` 和 `row_id` 打包进一个 64 位整数,使得按整数排序等于"先按分区、再按行号"排序。(b) 给定每行所属的分区号,不用比较排序,O(N) 地算出按分区分组后的行下标排列。

**常见追问:**
- 为什么要打包?(排序时只比较一个整数,省内存、缓存友好)
- 位数怎么分?(分区位数决定最大分区数,行号位数决定一个批里最大行数,两者要互相约束)
- 为什么计数排序比比较排序好?(分区号范围小且已知,O(N + P))
- 计数排序是稳定的吗?(是,同分区内保持原行序)

**Python Solution:**

```python
PARTITION_BITS = 24
ROW_BITS = 40  # illustrative split of 64 bits; real engines choose their own
ROW_MASK = (1 << ROW_BITS) - 1


def pack(partition_id, row_id):
    assert 0 <= partition_id < (1 << PARTITION_BITS)
    assert 0 <= row_id < (1 << ROW_BITS)
    return (partition_id << ROW_BITS) | row_id


def unpack(packed):
    return packed >> ROW_BITS, packed & ROW_MASK


def counting_sort_by_partition(partition_ids, num_partitions):
    """Return (order, offsets): row indices grouped by partition, stable inside a group."""
    counts = [0] * num_partitions
    for p in partition_ids:
        counts[p] += 1
    offsets = [0] * (num_partitions + 1)
    for p in range(num_partitions):
        offsets[p + 1] = offsets[p] + counts[p]
    cursor = offsets[:-1].copy()
    order = [0] * len(partition_ids)
    for row, p in enumerate(partition_ids):
        order[cursor[p]] = row
        cursor[p] += 1
    return order, offsets


assert unpack(pack(5, 123)) == (5, 123)
assert unpack(pack(0, 0)) == (0, 0)
assert unpack(pack((1 << 24) - 1, (1 << 40) - 1)) == ((1 << 24) - 1, (1 << 40) - 1)

pids = [2, 0, 1, 0, 2, 1, 0]
packed = sorted(pack(p, r) for r, p in enumerate(pids))
assert [unpack(x) for x in packed] == [(0, 1), (0, 3), (0, 6), (1, 2), (1, 5), (2, 0), (2, 4)]

order, offsets = counting_sort_by_partition(pids, 3)
assert order == [1, 3, 6, 2, 5, 0, 4]  # same row order as the packed sort
assert offsets == [0, 3, 5, 7]
assert order == [unpack(x)[1] for x in packed]

assert counting_sort_by_partition([], 3) == ([], [0, 0, 0, 0])
order, offsets = counting_sort_by_partition([1, 1, 1], 2)
assert order == [0, 1, 2] and offsets == [0, 0, 3]
```

**复杂度:** 打包排序 O(N log N),计数排序 O(N + P)。
**和引擎的联系:** `[Source]` Gluten 的 `VeloxSortShuffleWriter` 就是把分区号和行号压进一个紧凑整数再排序;`getShuffleWriterType` 在分区数较多时选 sort shuffle 而不是 hash shuffle。具体的位数分配以源码为准,这里的 24/40 只是示意。

---

### Q11. 有界阻塞队列(生产者-消费者)

**题目:** 实现线程安全的有界队列:队列满时 `put` 阻塞,空时 `get` 阻塞;支持 `close()`,关闭后 `get` 在取完数据后返回结束标记。

**常见追问:**
- 为什么要用 `while` 而不是 `if` 检查条件?(防止虚假唤醒,以及多个等待者被同时唤醒后条件已变)
- 为什么用 `notify` 而不是 `notify_all`?(这里满足条件的等待者只需一个;关闭时要 `notify_all`)
- Python 的 GIL 对这类代码有影响吗?(对正确性没有影响,仍然需要锁;GIL 只限制 CPU 并行)
- 这和背压的关系?(有界队列满了会阻塞上游,就是最简单的背压)

**Python Solution:**

```python
import threading
from collections import deque


class QueueClosed(Exception):
    pass


class BoundedQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.items = deque()
        self.closed = False
        lock = threading.Lock()
        self.not_full = threading.Condition(lock)
        self.not_empty = threading.Condition(lock)

    def put(self, item):
        with self.not_full:
            while len(self.items) >= self.capacity and not self.closed:
                self.not_full.wait()
            if self.closed:
                raise QueueClosed()
            self.items.append(item)
            self.not_empty.notify()

    def get(self):
        with self.not_empty:
            while not self.items and not self.closed:
                self.not_empty.wait()
            if not self.items:  # closed and drained
                raise QueueClosed()
            item = self.items.popleft()
            self.not_full.notify()
            return item

    def close(self):
        with self.not_empty:  # same underlying lock as not_full
            self.closed = True
            self.not_empty.notify_all()
            self.not_full.notify_all()


q = BoundedQueue(capacity=3)
results = []
results_lock = threading.Lock()


def producer(start):
    for i in range(start, start + 500):
        q.put(i)


def consumer():
    while True:
        try:
            item = q.get()
        except QueueClosed:
            return
        with results_lock:
            results.append(item)


producers = [threading.Thread(target=producer, args=(s,)) for s in (0, 1000)]
consumers = [threading.Thread(target=consumer) for _ in range(3)]
for t in producers + consumers:
    t.start()
for t in producers:
    t.join()
q.close()  # all items were put; consumers drain the rest and then exit
for t in consumers:
    t.join()

assert sorted(results) == list(range(0, 500)) + list(range(1000, 1500))
assert len(q.items) == 0 and q.closed

closed_q = BoundedQueue(1)
closed_q.close()
try:
    closed_q.put(1)
    raise AssertionError("put after close must fail")
except QueueClosed:
    pass
try:
    closed_q.get()
    raise AssertionError("get on a closed, empty queue must fail")
except QueueClosed:
    pass
```

**复杂度:** 每次操作 O(1);阻塞时不占 CPU。
**和引擎的联系:** `[Source]` Ballista 的 `ShuffleWriterExec` 用容量为 8 的有界 channel(`DEFAULT_SHUFFLE_CHANNEL_CAPACITY`)把异步计算和阻塞 IO 隔开;Sail 的任务流也有 `task_stream_buffer`(默认 16)。有界缓冲 = 背压。

---

### Q12. 布隆过滤器(Bloom Filter)

**题目:** 实现布隆过滤器,给定预期元素数 `n` 和目标误判率 `p`,自动计算位数和哈希函数个数。

**常见追问:**
- 公式?(`m = -n·ln(p) / (ln 2)²`,`k = (m/n)·ln 2`)
- 能删除吗?(标准版不能;Counting Bloom Filter 可以)
- 引擎里用在哪?(runtime filter:把 build 侧的 key 做成 bloom filter 下推到 probe 侧的 scan,提前过滤掉大部分不匹配的行)
- 如何用一个哈希模拟 k 个?(双重哈希:`h_i = h1 + i·h2`)

**Python Solution:**

```python
import hashlib
import math


class BloomFilter:
    def __init__(self, expected_items, false_positive_rate):
        n, p = max(1, expected_items), false_positive_rate
        self.num_bits = max(8, int(math.ceil(-n * math.log(p) / (math.log(2) ** 2))))
        self.num_hashes = max(1, int(round(self.num_bits / n * math.log(2))))
        self.bits = bytearray((self.num_bits + 7) // 8)

    def _positions(self, item):
        digest = hashlib.blake2b(repr(item).encode("utf-8"), digest_size=16).digest()
        h1 = int.from_bytes(digest[:8], "little")
        h2 = int.from_bytes(digest[8:], "little") | 1  # odd -> better spread
        return [(h1 + i * h2) % self.num_bits for i in range(self.num_hashes)]

    def add(self, item):
        for pos in self._positions(item):
            self.bits[pos >> 3] |= 1 << (pos & 7)

    def might_contain(self, item):
        return all(self.bits[pos >> 3] & (1 << (pos & 7)) for pos in self._positions(item))


bf = BloomFilter(expected_items=1_000, false_positive_rate=0.01)
members = [f"user-{i}" for i in range(1_000)]
for m in members:
    bf.add(m)

assert all(bf.might_contain(m) for m in members)  # never a false negative

outsiders = [f"other-{i}" for i in range(10_000)]
false_positives = sum(1 for o in outsiders if bf.might_contain(o))
assert false_positives / len(outsiders) < 0.03  # target is 1%; leave headroom

assert 9 * 1_000 < bf.num_bits < 11 * 1_000  # ~9.6 bits per element at p = 1%
assert bf.num_hashes in (6, 7)
assert not BloomFilter(10, 0.01).might_contain("anything")  # empty filter
```

**复杂度:** `add` 和 `might_contain` 都是 O(k)。
**和引擎的联系:** `[Source]` Gluten 的 Velox 后端注册了 `BloomFilterMightContainJointRewriteRule`,Spark 的 runtime bloom filter 也是靠它转成原生执行。这是 join 优化里很常见的话题。

---

### Q13. 合并读取范围(I/O 合并)

**题目:** 给一批待读取的字节区间 `(start, end)`,把距离不超过 `max_gap` 的区间合并成一次读取,合并后的单次读取不能超过 `max_size`。

**常见追问:**
- 为什么要合并?(对象存储每次请求有固定延迟,读几个小区间不如读一个稍大的区间)
- 合并会多读无用字节,怎么权衡?(`max_gap` 就是"多读多少字节换一次请求")
- 区间重叠怎么处理?(合并成它们的并集)

**Python Solution:**

```python
def coalesce_ranges(ranges, max_gap, max_size=None):
    """Merge [start, end) ranges whose gap is <= max_gap; cap merged length at max_size."""
    if not ranges:
        return []
    ordered = sorted(ranges)
    merged = [list(ordered[0])]
    for start, end in ordered[1:]:
        cur = merged[-1]
        gap = start - cur[1]
        new_end = max(cur[1], end)
        fits = max_size is None or new_end - cur[0] <= max_size
        if gap <= max_gap and fits:
            cur[1] = new_end
        else:
            merged.append([start, end])
    return [tuple(r) for r in merged]


assert coalesce_ranges([], 10) == []
assert coalesce_ranges([(0, 10)], 0) == [(0, 10)]
assert coalesce_ranges([(0, 10), (12, 20)], max_gap=2) == [(0, 20)]  # gap of 2 is merged
assert coalesce_ranges([(0, 10), (13, 20)], max_gap=2) == [(0, 10), (13, 20)]  # gap of 3 is not
assert coalesce_ranges([(5, 15), (0, 10)], max_gap=0) == [(0, 15)]  # unsorted and overlapping
assert coalesce_ranges([(0, 10), (0, 10)], max_gap=0) == [(0, 10)]  # duplicates
assert coalesce_ranges([(0, 10), (10, 20), (20, 30)], 0, max_size=20) == [(0, 20), (20, 30)]

ranges = [(0, 4), (6, 9), (100, 110), (112, 120), (500, 501)]
assert coalesce_ranges(ranges, max_gap=5) == [(0, 9), (100, 120), (500, 501)]
```

**复杂度:** 时间 O(R log R),空间 O(R)。
**和引擎的联系:** Parquet 读取时,列块(column chunk)的字节范围就是这样合并后再发给对象存储的。这是存储层和数据湖引擎的基本功。

---

## 3. 口头概念题(写完代码后常被追问)

### C1. Hash join、sort-merge join、broadcast join 怎么选?为什么 Spark 默认倾向 sort-merge?

**回答要点:**
- `[General]` **Broadcast hash join:** 一侧很小(小于广播阈值),广播到所有节点,没有 shuffle。风险:广播表过大会让 driver 和 executor OOM;统计不准会误判。
- `[General]` **Shuffled hash join:** 两侧按 key shuffle,然后在每个分区里对较小的一侧建 hash 表。快,但 build 侧必须放进内存;倾斜时单个分区的 hash 表会很大。
- `[General]` **Sort-merge join:** 两侧 shuffle 后各自排序再合并。稳健,内存可控(排序可以 spill),对数据规模不敏感,所以 Spark 默认 `preferSortMergeJoin=true`;代价是排序的 CPU 开销。
- `[General]` 没有 equi 条件时只能用 nested loop(BNLJ)或 Cartesian,复杂度 O(n·m)。
- `[Source]` **Gluten 的做法**(本地 `~/github/gluten`,`main` @ `7b7ff7c92`):策略仍由 Spark 的 `JoinSelection` 决定,Gluten 的 `OffloadJoin` 一一换成 Transformer。唯一的例外是 `RewriteJoin`:`spark.gluten.sql.columnar.forceShuffledHashJoin` 默认 `true`,会把 `SortMergeJoinExec` 改写成 `ShuffledHashJoinExec`。build 侧选择:先看 join 类型是否允许在某一侧建表;两侧都可以时用 `getOptimalBuildSide`(逻辑计划的 `stats.sizeInBytes`,取小的一侧);没有逻辑链接就默认 BuildLeft。
- `[Source]` **原生侧:** Velox 后端把 `isSMJ=` 转成 `MergeJoinNode`,其他等值 join 转成 `HashJoinNode`,BNLJ 和 Cartesian 转成 `NestedLoopJoinNode`。
- 原因推断(`[General]`):Velox 的 hash join 是向量化的,而且支持 spill,所以 Gluten 敢默认用 hash join,抵消了 Spark 里 SHJ 容易 OOM 的缺点。

**常见追问:** build side 选错了会怎样?(hash 表过大,spill 或 OOM,性能骤降。解决:提高统计质量、开启 AQE、必要时关闭 `forceShuffledHashJoin`)

---

### C2. Pipelined shuffle 和 blocking shuffle 的区别?Hybrid 是什么?

**回答要点:**
- `[General]` **Pipelined:** 上游算出来的数据直接流给下游,上下游同时运行,延迟低,数据只在内存里。缺点:所有 task 要同时有资源,任意一个失败常常要重跑整个相关区域。
- `[General]` **Blocking:** 上游先把数据完整写出(通常到磁盘),下游再启动读取。资源需求低、可以按分区单独重试、可以拿到完整的统计信息做自适应优化。缺点:有 stage 屏障,延迟高,多一次落盘。
- `[General]` **Hybrid(如 Flink 的 Hybrid Shuffle):** 内存优先,压力大时落盘,下游可以在上游还没结束时开始读,屏障被去掉但保留磁盘兜底。
- `[Source]` **Sail:** 默认后端 `flight` 是 pipelined(`SAIL_CLUSTER__SHUFFLE_BACKEND__TYPE=flight`);`storage` 和 `celeborn` 后端是 blocking。官方文档说 pipelined 失败时整个 job 要重试。
- `[Source]` **Ballista:** 上游 stage 写完 Arrow IPC 文件后,下游通过 Flight 或 `IO_BLOCK_TRANSPORT` 拉取,本质是 blocking。

**常见追问:** 什么时候选 pipelined,什么时候选 blocking?(小数据、资源足、要低延迟用 pipelined;大数据、需要自适应合并或倾斜处理、失败率高用 blocking)

---

### C3. Hash shuffle 和 sort shuffle 的区别?

**回答要点:**
- `[General]` **Hash shuffle:** 每个 map task 为每个 reduce 分区写一个文件,文件数 N×M,分区多时小文件爆炸。
- `[General]` **Sort shuffle:** 每个 map task 按目标分区排序后写**一个数据文件加一个索引文件**,文件数 2×N,reduce 端按索引读自己的字节范围。
- `[Source]` Ballista 的 `sort_shuffle/` 模块文档说得很明确:从 N×M 降到 2×N,思路来自 Spark。
- `[Source]` Gluten 的 `ColumnarShuffleManager` 继承 Spark 的 `SortShuffleManager`;`getShuffleWriterType` 在分区数或列数很大时选 sort shuffle(我之前的指南里记录的阈值是分区数 ≥ 4000 或列数 ≥ 100000,以源码为准)。
- `[General]` 取舍:hash shuffle 写入快、不需要排序,但文件多;sort shuffle 文件少、对大分区数友好,但多一步排序或计数分组。

---

### C4. 内存不够时引擎怎么办?(spill 与内存仲裁)

**回答要点:**
- `[General]` 每个算子向内存池申请预算;池满时由仲裁器让某些算子 spill(把状态写盘并释放内存)。
- `[General]` 选谁 spill:通常是占用最大的,这样一次 spill 释放最多。
- `[Source]` Gluten 的内存栈(详见 `datapelago_gluten_shuffle_memory_guide.md`):`TreeMemoryConsumer` 组成树;`TreeMemoryTargets.spillTree` 先 SHRINK 再 SPILL,最大者优先;`ThrowOnOomMemoryTarget` 在抛 OOM 前先 GC,并最多重试 9 次、总共睡眠不超过 1 秒;`OverAcquire` 以 0.3 的比例多申请一点余量;native 侧由 `ListenableArbitrator` 回调到 JVM 的仲裁器。
- `[General]` 难点:spill 本身不能再申请内存;spill 期间的并发申请;spill 后要能继续读取(外部排序的归并读)。

**常见追问:** 为什么 shuffle writer 的 spill 和算子 spill 要分开处理?(shuffle 缓冲是写出路径的一部分,可以更激进地换出;算子状态换出后还要读回来)

---

### C5. 数据倾斜怎么处理?

**回答要点:**
- `[General]` **检测:** 分区大小大于中位数的若干倍(如 5 倍),并且超过一个绝对下限(如 128 MiB),避免对很小的数据误报。
- `[General]` **缓解:** (a) 拆分倾斜分区,让多个 task 分担,另一侧对应数据复制到每个 task(skew join);(b) 对倾斜 key 加盐(salting)再两阶段聚合;(c) 把小表广播,绕开 shuffle;(d) 过滤掉异常 key(例如 null)单独处理。
- `[Source]` 你在 Sail 分支里做过:`a588d21` 拆分 partitioned join 里的倾斜分区、`503705d` 运行时把 partitioned hash join 转成 broadcast join。检测靠 `StageShuffleStats` 里各 channel 的字节数。这些都需要 blocking shuffle 才能拿到完整统计。
- `[General]` 自适应合并和倾斜处理要求上游完成,因此和"去掉 stage 屏障"是互相制约的。

---

### C6. 为什么列式、向量化执行更快?

**回答要点:**
- `[General]` 一次处理一批值,摊薄虚函数调用和分支预测失败;同类型数据连续存放,CPU 缓存和 SIMD 友好;列式存储只读用到的列,压缩率高。
- `[General]` 代价:行级的复杂逻辑(例如大量分支的 UDF)不容易向量化;和行式引擎交互需要行列转换(Gluten 里的 columnar-to-row、row-to-columnar 转换就是这个代价)。
- `[General]` Python UDF 是典型的慢路径:逐行或逐批跨语言调用。能用内置表达式就不用 UDF;必须用就用 pandas/Arrow 向量化 UDF。

---

### C7. Python 的 GIL 对数据处理代码有什么影响?

**回答要点:**
- `[General]` GIL 让同一时刻只有一个线程执行 Python 字节码,CPU 密集的多线程代码不能并行;IO 密集的多线程仍然有效,因为等待 IO 时会释放 GIL。
- `[General]` CPU 密集用多进程(`multiprocessing`、`concurrent.futures.ProcessPoolExecutor`)或把热点放进释放 GIL 的原生扩展(numpy、pyarrow)。
- `[General]` GIL 不保证复合操作的原子性,共享状态仍然要加锁(Q11)。
- `[General]` 新版本 Python 有可选的无 GIL 构建(free-threaded),但是否可用要看环境,别在面试里下断言。

---

### C8. 这些"Spark 加速"项目怎么区分?(Gluten、Comet、Auron、Ballista、Sail)

| 项目 | 定位 | 执行引擎 | 与 Spark 的关系 |
|---|---|---|---|
| Apache Gluten | Spark 插件 | 可换后端:Velox、ClickHouse、Bolt | 把 Spark 物理计划转成 Substrait 交给原生引擎 |
| DataFusion Comet | Spark 插件 | DataFusion(Rust) | 把 Spark 物理计划转成自己的 protobuf |
| Apache Auron(原 Blaze) | Spark 插件 | DataFusion(Rust) | 快手开源,同类思路 |
| DataFusion Ballista | 独立分布式引擎 | DataFusion(Rust) | 不依赖 Spark |
| LakeSail Sail | 独立引擎,Spark Connect 兼容 | DataFusion(Rust) | PySpark 客户端可直连 |

- `[Source]` Gluten 的后端目录在本地 `~/github/gluten`:`backends-velox`、`backends-clickhouse`、`backends-bolt`。
- `[Source]` DataPelago 的架构文档(`architecture.html`)写到 "converts any query or execution plan into standards-based formats like Substrait using mechanisms like Apache Gluten"。它没有说明 Nucleus 用什么引擎执行,也没有提到 Rust。
- `[Ask]` Nucleus 是否复用 Gluten 的 shuffle 和内存代码?执行引擎用什么语言?

---

### C9. Shuffle 数据丢了怎么办?(容错)

**回答要点:**
- `[General]` Shuffle 文件放在 executor 本地磁盘时,executor 挂了数据就丢,只能重算上游 stage 中丢失的分区。
- `[General]` 外置 shuffle 服务(如 Apache Celeborn)把数据放在独立集群并做副本,executor 挂了不影响已写入的数据,代价是多一跳网络和运维成本。
- `[Source]` Sail 已经有原生 Rust 的 Celeborn 客户端(`crates/sail-celeborn`),说的是 Celeborn 的 Netty 二进制帧加 protobuf 协议,配置 `SAIL_CLUSTER__SHUFFLE_BACKEND__TYPE=celeborn`。
- `[General]` 提交点:输出对下游"可见"之前必须已落盘;否则进程被 OOM kill 后无法恢复。Sail 的磁盘 shuffle 用"先写临时文件,再原子 rename"加"索引文件最后出现表示完整"来实现这个提交点。

---

### C10. Spark Connect 对这类插件有什么影响?

**回答要点:**
- `[General]` Spark Connect 是 Spark 自带的 gRPC 前端:客户端把计划序列化后发给 server,server 端翻译成 Catalyst 计划再走正常执行。
- `[General]` 因此 Gluten、Comet 这类挂在物理计划层的插件对 Connect **基本透明**,不需要为 Connect 单独改造。需要验证的是 Connect 特有路径(客户端 UDF、Arrow 结果回传、多 session 共享 JVM 时的全局状态)。
- `[Source]` 想让一个**独立**引擎直接说 Spark Connect,工作量就大得多:要实现 gRPC 服务、把 Connect 的关系和表达式翻译成引擎自己的计划。Sail 做了这件事;Ballista 在 issue #2249 里还在讨论。

---

## 4. Python 的坑(写代码时避开)

| 坑 | 说明 |
|---|---|
| 内置 `hash()` | 字符串哈希每个进程不同(哈希随机化),用于分区时必须换稳定哈希(如 `zlib.crc32`) |
| 可变默认参数 | `def f(x=[])` 的列表会在多次调用间共享 |
| 迭代时修改字典 | 会抛 `RuntimeError` |
| 浅拷贝 | `list.copy()`、`dict.copy()` 不会复制内部对象 |
| 比较 `None` 和数值 | `None < 1` 会抛 `TypeError`,排序混合类型要提供 `key` |
| 递归深度 | 默认约 1000 层,深递归改迭代 |
| 生成器只能消费一次 | 把它传给两个消费者会让第二个拿到空 |
| 浮点比较 | `0.1 + 0.2 != 0.3`,用容差或 `decimal` |
| `heapq` 是最小堆 | 要最大堆就取负,或者像 Q4 那样用 `(key, seq, item)` |
| 大整数不会溢出 | 位运算题要自己模拟 64 位(掩码 `& 0xFFFFFFFFFFFFFFFF`) |

---

## 5. 可以反问面试官的问题

这些都是 `[Ask]`,我没有公开来源能回答它们:

1. Nucleus 的执行引擎用什么语言实现?和 Spark 之间的接口是 Substrait 吗?
2. Shuffle 和内存管理是复用 Gluten / Velox 的实现,还是自研的?
3. 这个岗位日常是写引擎内核、写 Spark 集成层,还是写工具链和测试?
4. 团队怎么验证正确性和性能(TPC-DS、回归基准、与 Spark 结果对拍)?
5. 遇到不支持的算子时,回退到 Spark 的粒度是怎样的?回退的代价怎么衡量?
6. 内存不足或 executor OOM 时,Nucleus 的行为是什么?有没有外置 shuffle 的计划?

---

## 6. 面试当天清单

**前一晚**
- 用 Python 各手写一遍并跑通:Q1(hash join)、Q2(k 路归并)、Q5(LRU)。
- 口头讲一遍 C1(join 选择)和 C4(spill)。
- 确认编辑器 / 环境:能否运行代码、有没有自动补全、能不能用标准库。

**开场**
- 先问语言是否可以自选,是否能运行和写测试。
- 复述题目和约束,问清楚输入规模。

**写题时**
- 先说思路和复杂度,再写代码;写函数签名和一两个例子。
- 写完用空输入、单行、重复、null、大输入各测一遍。
- 主动说"如果内存不够 / 要并行 / 要向量化,我会这样改"。

**结束时**
- 总结复杂度和已知限制。
- 问 1–2 个第 5 节的问题。

---

## 7. 本文的依据与限制

- 第 2 节的 13 段 Python 代码在仓库里有测试(`tests/test_datapelago_coding_guide.py`),测试会真实执行每段代码和里面的断言。
- `[Source]` 标记的内容来自我本地读过的代码:`~/github/gluten`(`main` @ `7b7ff7c92`,2026-10-30)、`~/github/datafusion-ballista`(2026-05-14)、`~/github/sail`(`origin/main` @ `553fbdb73`,2026-10-28)以及你的 Sail 分支 `fork/disk-shuffle-integration`。我只读了代码,没有在这些项目上运行过基准。
- `[General]` 标记的是通用知识,没有逐条核对源码。
- 面试题本身是推测。如果面试官出了不同方向的题(例如纯算法题、系统设计、SQL),请以现场为准。
