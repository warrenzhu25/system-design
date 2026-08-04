# Microsoft Interview Questions

Problems reported from Microsoft loops, with emphasis on the MAI (Microsoft AI) / Copilot
platform rounds. Every coding solution below is implemented and tested in Python. The
system-design round carries a tested Python reference for its algorithmic core.

---

## Table of Contents

1. [DNA Shotgun Sequencing](#1-dna-shotgun-sequencing)
2. [In-Memory SQL Engine](#2-in-memory-sql-engine)
3. [LRU Cache (LC 146) + Multithreading Variant](#3-lru-cache-lc-146--multithreading-variant)
4. [Top-K Largest Elements (Retain / Rank Stores)](#4-top-k-largest-elements-retain--rank-stores)
5. [Rate Limiter (Design + Implementation)](#5-rate-limiter-design--implementation)
6. [Job Scheduler / ETL Pipeline System Design](#6-job-scheduler--etl-pipeline-system-design)

---

## 1. DNA Shotgun Sequencing

| | |
|---|---|
| **Tracks** | MLE · SWE |
| **Tags** | graph · hard · string-processing · dfs |
| **Frequency** | Medium |
| **Last asked** | 2026-05-28 |
| **Stage** | phone-screen · onsite-coding |

Reconstruct a DNA string from fragments tagged at both ends. Three-part: ordered chain →
undirected chain (Eulerian path) → multi-chain decomposition. The signature MAI "platform"
coding problem.

**Problem Statement:**

Each input fragment is a `Sequence(start_id, end_id, payload)` where `start_id` and `end_id`
are short tag strings (e.g. `"AAA"`, `"AAC"`) and `payload` is an arbitrary string
contributing to the reconstructed DNA. All tags within a part are unique unless the prompt
explicitly allows reuse.

### Part 1 — Directed chain

A fragment's `end_id` equals the next fragment's `start_id`. There is exactly one valid
ordering using all fragments. Implement:

```
String shotgunSequence(List<Sequence> sequences)
```

Concatenate the payloads in the recovered order.

```
[("AAA","AAC","AAAA"),
 ("AGG","ACC","GGGG"),
 ("AAC","ACT","TTTT"),
 ("ACT","AGG","CCCC")]
→ "AAAATTTTCCCCGGGG"
```

### Part 2 — Undirected chain (Eulerian path)

The two tags on each fragment are no longer labelled start / end. A fragment can be traversed
in either direction, and two fragments connect whenever they share any tag. A valid traversal
still exists and still uses every fragment. Same return type — the assembled payload string.

```
[("A","B","AAAA"),
 ("B","C","TTTT"),
 ("C","D","CCCC"),
 ("D","B","GGGG")]
A → B → C → D → B
→ "AAAATTTTCCCCGGGG"
```

### Part 3 — Multi-chain decomposition

Input fragments may form multiple disjoint chains (each fragment still has a unique direction
within its chain, but no global order exists). Return every assembled chain. Several
candidates also report a variant where the prompt asks you to detect whether a clean
decomposition exists at all and surface ambiguous fragments separately.

**Test Cases:**

| Part | Input | Output |
|------|-------|--------|
| 1 | `[(AAA,AAC,AAAA),(AGG,ACC,GGGG),(AAC,ACT,TTTT),(ACT,AGG,CCCC)]` | `AAAATTTTCCCCGGGG` |
| 1 | `[(A,B,XY)]` | `XY` |
| 1 | `[]` | `""` |
| 1 | `[(A,B,X),(C,D,Y)]` | raises — two heads, not one chain |
| 1 | `[(A,B,X),(B,A,Y)]` | raises — pure cycle, no head |
| 2 | `[(A,B,AAAA),(B,C,TTTT),(C,D,CCCC),(D,B,GGGG)]` | `AAAATTTTCCCCGGGG` |
| 2 | `[(A,B,AAAA),(C,B,TTTG)]` | `AAAAGTTT` — second edge traversed B→C, payload reversed |
| 2 | `[(A,B,AB),(B,C,BC),(C,A,CA)]` | `ABBCCA` — all degrees even, start anywhere |
| 2 | `[(A,A,GG)]` | `GG` — self-loop |
| 2 | `[(A,B,GT),(A,B,GT)]` | `GTTG` — duplicate fragment is a second parallel edge |
| 2 | `[(A,B,X),(C,D,Y)]` | raises — disconnected |
| 3 | `[(AAA,AAC,AAAA),(TGG,TGA,CCCC),(AAC,ACT,TTTT),(TGA,TAA,GGGG)]` | `["AAAATTTT","CCCCGGGG"]` |
| 3 | `[(A,B,X),(A,C,Y),(P,Q,Z)]` | chains `["Z"]`, ambiguous `[X, Y]` — `A` branches |
| 3 | `[(A,B,X),(B,A,Y)]` | chains `[]`, ambiguous `[X, Y]` — cycle has no head |

**Key Insights:**

1. **Part 1** reduces to walking a directed multigraph where every node has
   in-degree = out-degree = 1 along the unique path: build a `start_id → fragment` index, find
   the head (the `start_id` that never appears as any `end_id`), then chase pointers in O(N).
2. **Part 2** is the canonical Eulerian path on an *undirected* multigraph: each tag becomes a
   vertex, each fragment becomes an undirected edge carrying the payload. A valid traversal
   exists when exactly zero or two vertices have odd degree; start from one of the odd-degree
   vertices (or any vertex if all are even) and run Hierholzer's algorithm, splicing sub-cycles
   into the main path.
3. When walking the Part 2 path, emit `payload` if you traverse the edge in its declared
   `tag1 → tag2` direction and `reverse(payload)` if you traverse the other way — **interviewers
   do not always volunteer this, candidates have had to ask.**
4. Index edges by integer id, not by endpoint pair. That is what makes duplicate fragments
   (parallel edges) and self-loops fall out for free: mark `used[edge_id]` rather than
   deleting from an adjacency set.
5. Keep a per-vertex `ptr` cursor into the adjacency list so each edge is examined O(1)
   amortized times — without it Hierholzer degrades to O(V·E).
6. **Part 3** is connected-component decomposition (union-find over tags) followed by the
   Part 1 walk per component. Multi-chain detection is just "how many components produced a
   non-empty walk"; a component is *ambiguous* if any tag starts or ends two fragments, or if
   it has no head (a cycle).

**Complexity:** Part 1 O(N) time / O(N) space. Part 2 O(V + E) time / O(V + E) space with
iterative Hierholzer. Part 3 O(N·α(N)) for union-find plus O(N) for the per-component walks.

**Python Solution:**

```python
from collections import defaultdict
from dataclasses import dataclass


@dataclass(frozen=True)
class Sequence:
    start_id: str
    end_id: str
    payload: str


# ---------- Part 1: directed chain ----------
def shotgun_sequence(sequences: list[Sequence]) -> str:
    """
    Walk the unique directed chain. Time: O(N), Space: O(N)
    """
    if not sequences:
        return ""

    by_start: dict[str, Sequence] = {}
    for s in sequences:
        if s.start_id in by_start:
            raise ValueError(f"duplicate start tag {s.start_id!r}")
        by_start[s.start_id] = s

    # The head is the only start tag that is nobody's end tag.
    ends = {s.end_id for s in sequences}
    heads = [s.start_id for s in sequences if s.start_id not in ends]
    if len(heads) != 1:
        raise ValueError(f"expected exactly one head, found {len(heads)}")

    parts, tag, steps = [], heads[0], 0
    while tag in by_start and steps < len(sequences):
        s = by_start[tag]
        parts.append(s.payload)
        tag = s.end_id
        steps += 1
    if steps != len(sequences):        # a side cycle would strand fragments
        raise ValueError("fragments do not form a single chain")
    return "".join(parts)


# ---------- Part 2: undirected chain (Eulerian path) ----------
def shotgun_sequence_undirected(sequences: list[Sequence]) -> str:
    """
    Iterative Hierholzer over an undirected multigraph.
    Time: O(V + E), Space: O(V + E)
    """
    if not sequences:
        return ""

    # Each fragment is one undirected edge, identified by its index so that
    # duplicate fragments stay distinct parallel edges.
    adj: dict[str, list[tuple[str, int]]] = defaultdict(list)
    degree: dict[str, int] = defaultdict(int)
    order: list[str] = []              # first-seen tag order, for deterministic starts
    for i, s in enumerate(sequences):
        for t in (s.start_id, s.end_id):
            if t not in degree:
                order.append(t)
            degree[t] += 1
        adj[s.start_id].append((s.end_id, i))
        if s.start_id != s.end_id:     # a self-loop is stored once, degree counted twice
            adj[s.end_id].append((s.start_id, i))

    odd = [t for t in order if degree[t] % 2 == 1]
    if len(odd) not in (0, 2):
        raise ValueError(f"no Eulerian path: {len(odd)} odd-degree tags")
    start = odd[0] if odd else order[0]

    # Iterative Hierholzer: recursion blows the stack on long chains.
    ptr = {t: 0 for t in adj}          # per-vertex cursor keeps this O(E) overall
    used = [False] * len(sequences)
    stack: list[tuple[str, int]] = [(start, -1)]   # (vertex, edge used to arrive)
    path: list[tuple[str, int]] = []
    while stack:
        v, _ = stack[-1]
        lst = adj[v]
        while ptr[v] < len(lst) and used[lst[ptr[v]][1]]:
            ptr[v] += 1
        if ptr[v] == len(lst):
            path.append(stack.pop())   # vertex exhausted: splice it into the path
        else:
            u, eid = lst[ptr[v]]
            ptr[v] += 1
            used[eid] = True
            stack.append((u, eid))
    path.reverse()
    if len(path) != len(sequences) + 1:
        raise ValueError("disconnected: no single traversal uses every fragment")

    # Reverse the payload whenever the edge is walked against its declared direction.
    out = []
    for i in range(1, len(path)):
        prev = path[i - 1][0]
        _, eid = path[i]
        s = sequences[eid]
        out.append(s.payload if s.start_id == prev else s.payload[::-1])
    return "".join(out)


# ---------- Part 3: multi-chain decomposition ----------
def decompose_chains(sequences: list[Sequence]) -> tuple[list[str], list[Sequence]]:
    """
    Union-find components, then the Part 1 walk per component.
    Returns (assembled chains, fragments that do not form a clean chain).
    Time: O(N * alpha(N)), Space: O(N)
    """
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]       # path halving
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for s in sequences:
        union(s.start_id, s.end_id)

    groups: dict[str, list[int]] = defaultdict(list)
    for i, s in enumerate(sequences):
        groups[find(s.start_id)].append(i)

    chains: list[tuple[int, str]] = []
    ambiguous: list[int] = []
    for idxs in groups.values():
        by_start: dict[str, int] = {}
        end_count: dict[str, int] = defaultdict(int)
        conflict = False
        for i in idxs:
            s = sequences[i]
            if s.start_id in by_start:          # a tag starting two fragments branches
                conflict = True
            by_start[s.start_id] = i
            end_count[s.end_id] += 1
        if conflict or any(c > 1 for c in end_count.values()):
            ambiguous.extend(idxs)
            continue

        heads = [sequences[i].start_id for i in idxs
                 if sequences[i].start_id not in end_count]
        if len(heads) != 1:                     # 0 heads == cycle
            ambiguous.extend(idxs)
            continue

        head_idx = by_start[heads[0]]
        parts, tag, steps = [], heads[0], 0
        while tag in by_start and steps < len(idxs):
            s = sequences[by_start[tag]]
            parts.append(s.payload)
            tag = s.end_id
            steps += 1
        if steps != len(idxs):
            ambiguous.extend(idxs)
            continue
        chains.append((head_idx, "".join(parts)))

    chains.sort(key=lambda p: p[0])             # report in input order of each chain's head
    return [c for _, c in chains], [sequences[i] for i in sorted(ambiguous)]


def shotgun_sequence_multi(sequences: list[Sequence]) -> list[str]:
    """Strict variant: every fragment must belong to a clean chain."""
    chains, ambiguous = decompose_chains(sequences)
    if ambiguous:
        raise ValueError(f"{len(ambiguous)} fragment(s) do not form a clean chain")
    return chains


# Example usage
if __name__ == "__main__":
    print(shotgun_sequence([
        Sequence("AAA", "AAC", "AAAA"), Sequence("AGG", "ACC", "GGGG"),
        Sequence("AAC", "ACT", "TTTT"), Sequence("ACT", "AGG", "CCCC")]))
    # AAAATTTTCCCCGGGG

    print(shotgun_sequence_undirected([
        Sequence("A", "B", "AAAA"), Sequence("B", "C", "TTTT"),
        Sequence("C", "D", "CCCC"), Sequence("D", "B", "GGGG")]))
    # AAAATTTTCCCCGGGG

    print(shotgun_sequence_multi([
        Sequence("AAA", "AAC", "AAAA"), Sequence("TGG", "TGA", "CCCC"),
        Sequence("AAC", "ACT", "TTTT"), Sequence("TGA", "TAA", "GGGG")]))
    # ['AAAATTTT', 'CCCCGGGG']
```

## 2. In-Memory SQL Engine

| | |
|---|---|
| **Tracks** | MLE · SWE |
| **Tags** | data-structure · parsing · hard · sql · in-memory-database · state-machine |
| **Frequency** | Medium |
| **Last asked** | 2026-05-28 |
| **Stage** | onsite-coding |

Build a single-table in-memory database supporting `SELECT`, `WHERE`, and `ORDER BY` in 4-5
escalating follow-ups. A CSV string is the initialization input; you write the parser too.

**Problem Statement:**

Implement a small class that takes a CSV string at construction time and answers a series of
query calls. The interviewer drives 4 follow-ups in a single 60-minute round.

### Part 1 — Construction + SELECT

```python
db = InMemoryDB(csv_string)
db.select(["col_a", "col_b"])   # rows, in insertion order
```

The CSV header line names columns; subsequent lines are rows. Strings, quoted fields, and
escaped commas must round-trip — interviewers explicitly seed inputs with embedded commas,
doubled-quote escaping (fields containing literal quotes), and trailing-whitespace columns.
**Hand-roll the CSV parser; using a library is rejected.**

### Part 2 — WHERE

```python
db.select(cols, where=[("age", ">", 18), ("city", "==", "NYC")])
```

Predicates are conjunctive (AND). Comparisons on numeric-looking strings should coerce; type
mismatch raises.

### Part 3 — ORDER BY

```python
db.select(cols, where=..., order_by=[("age", "DESC"), ("name", "ASC")])
```

Composite sort key, stable on ties.

### Part 4 — Aggregations (longer variant)

`SUM`, `COUNT`, `AVG`, `MIN`, `MAX` over a column with optional `GROUP BY`.

### Part 5 — Update / Insert / Delete (occasional)

Insert a single row, update by `WHERE`, delete by `WHERE`. Recompute any derived ordering on
demand.

**Example:**

```
Key,location,weather,temperature,data
1,"Sunnyvale","sunny",100,"datetimestamp"
```

- Quoted cells (`"Sunnyvale"`, `"sunny"`) stay strings; an unquoted numeric cell (`100`) is
  parsed to `int` at load time.
- `select(["location","temperature"], where=[("temperature", ">", 50)])` returns only the
  projected columns of the rows matching every filter.

**Test Cases:**

Parser, against the adversarial inputs interviewers actually use:

| Input | Parsed cells |
|-------|--------------|
| `a,"b,c",d` | `["a", "b,c", "d"]` — embedded comma |
| `"he said ""hi"""` | `['he said "hi"']` — doubled-quote escaping |
| `a ,  b` | `["a", "b"]` — unquoted whitespace trimmed |
| `" a "` | `[" a "]` — whitespace inside quotes preserved |
| `"x\ny",z` | `["x\ny", "z"]` — newline inside quotes is not a row break |
| `a,,b` | `["a", "", "b"]` — empty field |
| `a,b\r\nc,d\r\n` | 2 rows — CRLF handled |

Queries, against the 4-row table in the example above (rows keyed 1-4, temperatures
100 / 48 / 30 / 72, locations Sunnyvale / Seattle / Redmond / Sunnyvale):

| Call | Output |
|------|--------|
| `select(["location","temperature"])[0]` | `{location: "Sunnyvale", temperature: 100}` |
| `select(["Key"], where=[("temperature",">",50)])` | `[{Key:1},{Key:4}]` |
| `select(["Key"], where=[("temperature",">",50),("location","==","Sunnyvale")])` | `[{Key:1},{Key:4}]` |
| `select(["Key"], where=[("temperature",">=","72")])` | `[{Key:1},{Key:4}]` — numeric string coerces |
| `select(["Key"], where=[("location",">",5)])` | raises — cross-type comparison |
| `select(["nope"])` | raises — unknown column |
| `select(["Key"], order_by=[("temperature","DESC")])` | `[1,4,2,3]` |
| `select(["Key"], order_by=[("location","ASC"),("temperature","DESC")])` | `[3,2,1,4]` |
| `aggregate([("COUNT","*"),("SUM","temperature")])` | `[{COUNT(*):4, SUM(temperature):250}]` |
| `aggregate([("AVG","temperature")], group_by=["location"])` | `Sunnyvale 86.0, Seattle 48.0, Redmond 30.0` |
| `update({"weather":"hot"}, where=[("temperature",">",90)])` | `1` |
| `delete(where=[("temperature","<",50)])` | `2` |
| `InMemoryDB("").select()` | `[]` |

**Key Insights:**

1. **Parse-time typing is part of the spec** in some variants: unquoted numeric cells load as
   `int`, quoted cells stay `str`, so coercion only matters on cross-type comparisons. The
   parser therefore has to report *whether a cell was quoted*, not just its text — that one
   extra bit is the whole design.
2. **The reported failure mode for every candidate who shipped this is running out of time on
   the CSV parser.** The inputs are intentionally adversarial and consume 15-20 minutes if you
   start from scratch in the room.
3. Hand-roll a stateful single-pass parser — `FIELD_START` / `IN_FIELD` / `IN_QUOTED` /
   `AFTER_CLOSING_QUOTE` — it is faster than regex and far easier to debug under pressure.
   Inside `IN_QUOTED`, commas and newlines are ordinary characters; that is why regex splitting
   fails.
4. Beyond parsing the shape is straightforward: rows as `list[dict]`, predicates as small
   lambdas, sort with `cmp_to_key` for stable multi-key ordering. `select` is a four-line
   pipeline: **filter → sort → project → return**.
5. Python's `sorted`, Java's `List.sort`, and JS `Array#sort` (ES2019+) are all stable, so a
   single composite comparator gives you the correct tie behavior for free — no need to sort
   once per key in reverse order.
6. For aggregation, hold one dict of `column → running_state` per group and finalize once at
   the end — don't materialize groups. `AVG` carries `(total, count)` so it needs no second pass.

**Complexity:** parse O(C) in input characters. `select` O(N·P) to filter, O(N log N · K) to
sort, O(N·M) to project. `aggregate` is a single O(N·A) streaming pass with O(G) state.

**Python Solution:**

```python
from functools import cmp_to_key
from typing import Optional, Union

Value = Union[int, float, str]


def parse_csv(text: str) -> list[list[tuple[str, bool]]]:
    """
    Single-pass state machine. Returns rows of (raw_text, was_quoted).
    The was_quoted bit is what drives parse-time typing.
    Time: O(C) in characters, Space: O(C)
    """
    FIELD_START, IN_FIELD, IN_QUOTED, AFTER_QUOTE = 0, 1, 2, 3
    rows: list[list[tuple[str, bool]]] = []
    row: list[tuple[str, bool]] = []
    buf: list[str] = []
    quoted = False
    state = FIELD_START

    def end_field():
        nonlocal buf, quoted, state
        raw = "".join(buf)
        row.append((raw if quoted else raw.strip(), quoted))   # trim only unquoted cells
        buf, quoted, state = [], False, FIELD_START

    def end_row():
        nonlocal row
        end_field()
        rows.append(row)
        row = []

    for ch in text:
        if state == FIELD_START:
            if ch == '"':
                quoted, state = True, IN_QUOTED
            elif ch == ',':
                end_field()
            elif ch == '\n':
                end_row()
            elif ch != '\r':
                buf.append(ch)
                state = IN_FIELD
        elif state == IN_FIELD:
            if ch == ',':
                end_field()
            elif ch == '\n':
                end_row()
            elif ch != '\r':
                buf.append(ch)
        elif state == IN_QUOTED:
            if ch == '"':
                state = AFTER_QUOTE
            else:
                buf.append(ch)          # commas and newlines are literal here
        else:                            # AFTER_QUOTE
            if ch == '"':
                buf.append('"')         # doubled quote -> one literal quote
                state = IN_QUOTED
            elif ch == ',':
                end_field()
            elif ch == '\n':
                end_row()
            # anything else after a closing quote is stray junk: ignore

    if buf or row or state != FIELD_START:
        end_row()
    return [r for r in rows if r != [("", False)]]      # drop the trailing blank line


def coerce(raw: str, was_quoted: bool) -> Value:
    """Quoted cells stay str; unquoted numeric-looking cells become int/float."""
    if was_quoted:
        return raw
    try:
        return int(raw)
    except ValueError:
        pass
    try:
        return float(raw)
    except ValueError:
        return raw


def _numeric(v: Value) -> Optional[float]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def compare(a: Value, b: Value) -> int:
    """Numeric when both sides look numeric, else lexicographic. Mixed -> TypeError."""
    na, nb = _numeric(a), _numeric(b)
    if na is not None and nb is not None:
        return (na > nb) - (na < nb)
    if isinstance(a, str) and isinstance(b, str):
        return (a > b) - (a < b)
    raise TypeError(f"cannot compare {a!r} ({type(a).__name__}) "
                    f"with {b!r} ({type(b).__name__})")


OPS = {
    "==": lambda c: c == 0,
    "!=": lambda c: c != 0,
    "<":  lambda c: c < 0,
    "<=": lambda c: c <= 0,
    ">":  lambda c: c > 0,
    ">=": lambda c: c >= 0,
}


class InMemoryDB:
    def __init__(self, csv_text: str):
        parsed = parse_csv(csv_text)
        if not parsed:
            self.columns: list[str] = []
            self.rows: list[dict[str, Value]] = []
            return
        self.columns = [raw for raw, _ in parsed[0]]
        self.rows = []
        for cells in parsed[1:]:
            if len(cells) != len(self.columns):
                raise ValueError(f"row has {len(cells)} cells, "
                                 f"header has {len(self.columns)}")
            self.rows.append({c: coerce(raw, q)
                              for c, (raw, q) in zip(self.columns, cells)})

    # ---------- Parts 1-3: filter -> sort -> project ----------
    def select(self, columns=None, where=None, order_by=None) -> list[dict[str, Value]]:
        cols = self.columns if columns is None else columns
        for c in cols:
            self._require(c)
        rows = [r for r in self.rows if self._matches(r, where)]
        rows = self._sorted(rows, order_by)
        return [{c: r[c] for c in cols} for r in rows]

    # ---------- Part 4: aggregation ----------
    def aggregate(self, aggregates, where=None, group_by=None) -> list[dict[str, Value]]:
        """
        aggregates: [("SUM","age"), ("COUNT","*")]. Single streaming pass —
        groups are never materialized, only their running state.
        """
        group_by = group_by or []
        for c in group_by:
            self._require(c)
        state: dict[tuple, dict] = {}
        order: list[tuple] = []
        for row in self.rows:
            if not self._matches(row, where):
                continue
            key = tuple(row[c] for c in group_by)
            if key not in state:
                state[key] = {i: None for i in range(len(aggregates))}
                order.append(key)
            acc = state[key]
            for i, (fn, col) in enumerate(aggregates):
                acc[i] = self._step(fn.upper(), acc[i], row, col)

        out = []
        for key in order:
            acc = state[key]
            rec: dict[str, Value] = dict(zip(group_by, key))
            for i, (fn, col) in enumerate(aggregates):
                rec[f"{fn.upper()}({col})"] = self._finalize(fn.upper(), acc[i])
            out.append(rec)
        return out

    def _step(self, fn, cur, row, col):
        if fn == "COUNT":
            if col != "*" and row.get(col) is None:
                return cur or 0
            return (cur or 0) + 1
        self._require(col)
        v = row[col]
        if fn in ("SUM", "AVG"):
            num = _numeric(v)
            if num is None:
                raise TypeError(f"{fn} over non-numeric value {v!r}")
            total, count = cur or (0.0, 0)      # AVG carries (total, count): no second pass
            return (total + num, count + 1)
        if fn == "MIN":
            return v if cur is None or compare(v, cur) < 0 else cur
        if fn == "MAX":
            return v if cur is None or compare(v, cur) > 0 else cur
        raise ValueError(f"unknown aggregate {fn}")

    @staticmethod
    def _finalize(fn, cur):
        if fn == "COUNT":
            return cur or 0
        if fn == "SUM":
            total = (cur or (0.0, 0))[0]
            return int(total) if total == int(total) else total
        if fn == "AVG":
            total, count = cur or (0.0, 0)
            return total / count if count else None
        return cur

    # ---------- Part 5: insert / update / delete ----------
    def insert(self, row: dict[str, Value]) -> None:
        unknown = set(row) - set(self.columns)
        if unknown:
            raise ValueError(f"unknown column(s): {sorted(unknown)}")
        self.rows.append({c: row.get(c, "") for c in self.columns})

    def update(self, assignments: dict[str, Value], where=None) -> int:
        for c in assignments:
            self._require(c)
        n = 0
        for row in self.rows:
            if self._matches(row, where):
                row.update(assignments)
                n += 1
        return n

    def delete(self, where=None) -> int:
        keep = [r for r in self.rows if not self._matches(r, where)]
        removed = len(self.rows) - len(keep)
        self.rows = keep
        return removed

    # ---------- helpers ----------
    def _require(self, col: str) -> None:
        if col not in self.columns:
            raise KeyError(f"unknown column {col!r}")

    def _matches(self, row, where) -> bool:
        for col, op, target in where or []:      # conjunctive: every predicate must hold
            self._require(col)
            if op not in OPS:
                raise ValueError(f"unknown operator {op!r}")
            if not OPS[op](compare(row[col], target)):
                return False
        return True

    def _sorted(self, rows, order_by):
        if not order_by:
            return rows
        for col, _ in order_by:
            self._require(col)

        def cmp(a, b):
            for col, direction in order_by:
                c = compare(a[col], b[col])
                if c:
                    return -c if direction.upper() == "DESC" else c
            return 0

        return sorted(rows, key=cmp_to_key(cmp))   # sorted() is stable -> ties keep input order


# Example usage
if __name__ == "__main__":
    csv_text = (
        'Key,location,weather,temperature,data\n'
        '1,"Sunnyvale","sunny",100,"datetimestamp"\n'
        '2,"Seattle","rain, heavy",48,"datetimestamp"\n'
        '3,"Redmond","he said ""cold""",30,"datetimestamp"\n'
        '4,"Sunnyvale","sunny",72,"datetimestamp"\n'
    )
    db = InMemoryDB(csv_text)

    print(db.select(["location", "temperature"], where=[("temperature", ">", 50)]))
    # [{'location': 'Sunnyvale', 'temperature': 100},
    #  {'location': 'Sunnyvale', 'temperature': 72}]

    print(db.select(["Key"], order_by=[("location", "ASC"), ("temperature", "DESC")]))
    # [{'Key': 3}, {'Key': 2}, {'Key': 1}, {'Key': 4}]

    print(db.aggregate([("AVG", "temperature"), ("COUNT", "*")], group_by=["location"]))
    # [{'location': 'Sunnyvale', 'AVG(temperature)': 86.0, 'COUNT(*)': 2},
    #  {'location': 'Seattle', ...}, {'location': 'Redmond', ...}]
```

## 3. LRU Cache (LC 146) + Multithreading Variant

| | |
|---|---|
| **Tracks** | SWE · MLE |
| **Tags** | linked-list · hashmap · data-structure · concurrency · medium |
| **Frequency** | Medium |
| **Last asked** | 2026-04-21 |
| **Stage** | phone-screen · onsite-coding |

The single most repeated Microsoft coding prompt. O(1) `get` / `put` bounded cache, with
frequent follow-ups on thread safety and LFU variants.

**Problem Statement:**

Implement an `LRUCache(capacity)` with O(1) `get(key)` and `put(key, value)`. On eviction,
drop the least-recently-used entry. `get` counts as a use.

Reported follow-ups, in order of frequency:

1. **Strict O(1).** Multiple interviewers explicitly reject `OrderedDict.move_to_end` based
   solutions on the grounds that internal reordering is O(log N) under Python's hash table
   assumption; the expected answer is hashmap + doubly-linked-list.
2. **Thread safety (MAI variant).** Multiple concurrent `get` / `put` callers; protect with a
   single `RLock`, or with reader-writer locks if performance is probed.
3. **LFU (one VO round):** same shape as LRU but evict the least-frequently-used; on frequency
   ties, fall back to LRU.

**Test Cases:**

| Operation sequence (capacity 2) | Result |
|---------------------------------|--------|
| `put(1,1); put(2,2); get(1)` | `1` |
| `put(3,3); get(2)` | `-1` — key 2 was LRU |
| `put(4,4); get(1)` | `-1` — key 1 evicted next |
| `get(3); get(4)` | `3`, `4` |
| `put(1,1); put(1,10); get(1)` + size | `10`, size stays `1` — update is not an insert |
| capacity 3: `put(1..3); get(1)` then dump MRU→LRU | `[1, 3, 2]` |
| capacity 1: `put(1,1); put(2,2)` | `get(1) == -1`, `get(2) == 2` |
| LFU capacity 2: `put(1,1); put(2,2); get(1); put(3,3)` | `get(2) == -1` — lowest freq evicted |
| LFU continued: `put(4,4)` (1 and 3 both at freq 2) | `get(1) == -1` — LRU breaks the tie |
| LFU capacity 0: `put(1,1)` | `get(1) == -1` |
| 8 threads × 2000 mixed `get`/`put` | no torn reads, size stays ≤ capacity, list length == map size |

**Key Insights:**

1. The canonical hashmap + doubly-linked-list pattern:
   - Dict maps `key → Node`.
   - Doubly-linked list ordered MRU → LRU. Head sentinel for MRU, tail sentinel for LRU.
   - `get(key)`: O(1) lookup, splice node to head.
   - `put(key, value)`: insert at head; if over capacity, unlink `tail.prev` and pop from dict.
2. Use sentinel nodes at both ends so every splice is a four-pointer reassignment with **no
   special cases** — no empty-list branch, no head/tail branch.
3. Evict *before* inserting, and only when the key is genuinely new. Updating an existing key
   must not evict.
4. For thread safety, the simplest correct answer is a single `threading.RLock` wrapping every
   public method. Interviewers who push on this expect you to acknowledge the contention cost
   and discuss **sharding** the cache (split into N independent shards by `hash(key) % N`, each
   with its own lock) — the same pattern as `ConcurrentHashMap`.
5. A reader-writer lock does *not* help a plain LRU: `get` mutates the recency list, so it is a
   writer too. Say this out loud — it is the trap in the "use an RWLock" follow-up. (It helps
   only if you relax recency tracking, e.g. sampled or clock-based eviction.)
6. LFU is LRU + a `freq → DLL` index: each `get` / `put` bumps the node from its current
   frequency bucket into the next; eviction targets the LRU node within the lowest-frequency
   bucket. Track the minimum frequency to find the eviction bucket in O(1). An insertion-ordered
   set (`LinkedHashSet` / Python `dict` / JS `Set`) gives you the intra-bucket LRU order for free.
7. `min_freq` only ever needs to move to `f + 1` when bucket `f` empties during a bump, and
   resets to `1` on every insert. That is the whole bookkeeping.

**Complexity:** all operations O(1) amortized, O(capacity) space, for both LRU and LFU.

**Python Solution:**

```python
import threading
from collections import defaultdict


class Node:
    __slots__ = ("key", "value", "prev", "next")

    def __init__(self, key=None, value=None):
        self.key = key
        self.value = value
        self.prev = None
        self.next = None


class LRUCache:
    """
    hashmap + doubly-linked list.
    head sentinel = MRU side, tail sentinel = LRU side.
    get/put: O(1). Space: O(capacity).
    """

    def __init__(self, capacity: int):
        if capacity <= 0:
            raise ValueError("capacity must be positive")
        self.capacity = capacity
        self.map: dict = {}
        self.head = Node()          # sentinels remove every edge case from splicing
        self.tail = Node()
        self.head.next = self.tail
        self.tail.prev = self.head

    def _unlink(self, node: Node) -> None:
        node.prev.next = node.next
        node.next.prev = node.prev

    def _push_front(self, node: Node) -> None:
        node.prev = self.head
        node.next = self.head.next
        self.head.next.prev = node
        self.head.next = node

    def get(self, key) -> int:
        node = self.map.get(key)
        if node is None:
            return -1
        self._unlink(node)          # a read counts as a use
        self._push_front(node)
        return node.value

    def put(self, key, value) -> None:
        node = self.map.get(key)
        if node is not None:        # update: refresh recency, never evict
            node.value = value
            self._unlink(node)
            self._push_front(node)
            return
        if len(self.map) == self.capacity:
            lru = self.tail.prev
            self._unlink(lru)
            del self.map[lru.key]
        node = Node(key, value)
        self.map[key] = node
        self._push_front(node)

    def keys_mru_to_lru(self):
        out, cur = [], self.head.next
        while cur is not self.tail:
            out.append(cur.key)
            cur = cur.next
        return out


class ThreadSafeLRUCache(LRUCache):
    """
    Follow-up 2: one RLock around every public method.
    Note that get() mutates the recency list, so it is a writer —
    a reader-writer lock buys nothing here.
    """

    def __init__(self, capacity: int):
        super().__init__(capacity)
        self._lock = threading.RLock()

    def get(self, key) -> int:
        with self._lock:
            return super().get(key)

    def put(self, key, value) -> None:
        with self._lock:
            super().put(key, value)


class ShardedLRUCache:
    """
    Follow-up 2b: N independent shards, each with its own lock
    (the ConcurrentHashMap pattern). Cuts contention ~N-fold.
    """

    def __init__(self, capacity: int, shards: int = 16):
        shards = max(1, min(shards, capacity))
        base, extra = divmod(capacity, shards)
        self.shards = [ThreadSafeLRUCache(base + (1 if i < extra else 0))
                       for i in range(shards)]

    def _shard(self, key):
        return self.shards[hash(key) % len(self.shards)]

    def get(self, key):
        return self._shard(key).get(key)

    def put(self, key, value):
        self._shard(key).put(key, value)


class LFUCache:
    """
    Follow-up 3: LFU — evict least frequently used, LRU within a frequency tie.
    A Python dict preserves insertion order, so each bucket is its own LRU queue.
    get/put: O(1). Space: O(capacity).
    """

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.values: dict = {}
        self.freq: dict = {}
        self.buckets: dict = defaultdict(dict)   # freq -> {key: None}, insertion-ordered
        self.min_freq = 0

    def _bump(self, key) -> None:
        f = self.freq[key]
        del self.buckets[f][key]
        if not self.buckets[f]:
            del self.buckets[f]
            if self.min_freq == f:               # the only way min_freq ever advances
                self.min_freq = f + 1
        self.freq[key] = f + 1
        self.buckets[f + 1][key] = None

    def get(self, key) -> int:
        if key not in self.values:
            return -1
        self._bump(key)
        return self.values[key]

    def put(self, key, value) -> None:
        if self.capacity <= 0:
            return
        if key in self.values:
            self.values[key] = value
            self._bump(key)
            return
        if len(self.values) == self.capacity:
            victim = next(iter(self.buckets[self.min_freq]))   # oldest in lowest bucket
            del self.buckets[self.min_freq][victim]
            if not self.buckets[self.min_freq]:
                del self.buckets[self.min_freq]
            del self.values[victim]
            del self.freq[victim]
        self.values[key] = value
        self.freq[key] = 1
        self.buckets[1][key] = None
        self.min_freq = 1                        # a fresh insert always resets min_freq


# Example usage
if __name__ == "__main__":
    cache = LRUCache(2)
    cache.put(1, 1)
    cache.put(2, 2)
    print(cache.get(1))   # 1
    cache.put(3, 3)       # evicts key 2
    print(cache.get(2))   # -1

    lfu = LFUCache(2)
    lfu.put(1, 1)
    lfu.put(2, 2)
    lfu.get(1)            # freq: 1 -> 2, 2 -> 1
    lfu.put(3, 3)         # evicts key 2
    print(lfu.get(2))     # -1
```

## 4. Top-K Largest Elements (Retain / Rank Stores)

| | |
|---|---|
| **Tracks** | MLE · SWE |
| **Tags** | heap · top-k · sorting · easy |
| **Frequency** | Low · New |
| **Last asked** | 2026-05-15 |
| **Stage** | phone-screen |

The recurring MAI phone-screen coding warm-up: return the top-K elements of a list under a
simple ranking rule. Appears in two shapes — retain the K largest in original order, or rank
business candidates by a composite key. Each runs ~15-20 minutes after the BQ section of a
45-minute MAI phone screen.

**Problem Statement:**

### Shape 1 — Retain the K largest, original order

```
input:  nums (list of ints), k (int)
output: the original list with only the K largest values kept; smaller values
        removed, remaining elements stay in their original order
```

### Shape 2 — Rank stores by a composite key

```
input:  list of business candidates, each with (score, distance, isOpen)
output: the top-K candidates sorted by score descending, ties broken by
        distance ascending
```

Clarify up front how `isOpen` is used — the prompt lists it as a field but does not always
state whether closed candidates are filtered out before ranking.

**Test Cases:**

Shape 1 (the tie-at-cutoff rule below is "keep the earliest occurrences", which is the
convention you should confirm before coding):

| Input | Output | Why |
|-------|--------|-----|
| `[3,1,5,2,4], k=2` | `[5,4]` | |
| `[3,1,5,2,4], k=3` | `[3,5,4]` | original order preserved, not sorted |
| `[4,1,4,3], k=3` | `[4,4,3]` | |
| `[5,5,5,1], k=2` | `[5,5]` | three elements tie at the cutoff; keep exactly K |
| `[2,2,2,2], k=2` | `[2,2]` | every element ties; the tie budget stops at K |
| `[-5,-1,-3,-2], k=2` | `[-1,-2]` | negatives |
| `[1,9,9,2], k=1` | `[9]` | first of the tied maxima |
| `[1,2,3], k=0` or `k=-1` | `[]` | |
| `[1,2,3], k=5` | `[1,2,3]` | k ≥ n returns everything |
| `[], k=3` | `[]` | |

Shape 2, against `A(4.5, 2.0, open)`, `B(4.9, 5.0, open)`, `C(4.5, 1.0, closed)`,
`D(3.0, 0.5, open)`, `E(4.9, 5.0, open)`:

| Call | Output | Why |
|------|--------|-----|
| `k=2` | `[B, E]` | highest score first |
| `k=4` | `[B, E, C, A]` | C before A: same score, nearer |
| `k=5` | `[B, E, C, A, D]` | B before E: identical keys, input order preserved |
| `k=3, open_only=True` | `[B, E, A]` | C is filtered out before ranking |
| `k=99` | all 5 | k > n |
| `k=0` | `[]` | |

**Key Insights:**

1. Both shapes are heap / partial-sort exercises. A size-K min-heap gives **O(N log K)**;
   `heapq.nlargest(k, items, key=...)` handles the composite-key shape directly.
2. **The whole difficulty is the duplicates-at-threshold case.** For Shape 1, if you find the
   K-th largest value and then sweep keeping everything `>= threshold`, several elements can
   tie at the cutoff and you return **more than K**. Two defensible rules — keep exactly K
   (earliest occurrences win) or keep all ties — and the round grades you for asking which one
   before coding. Both are implemented below.
3. The exact-K fix is a **tie budget**: count how many elements are strictly greater than the
   threshold, then allow only `k - that_count` elements equal to it during the sweep.
4. The heap approach gets the same tie rule for free by ordering on `(value, -index)`: among
   equal values the *later* index sorts smaller, so it is the one evicted, and the earliest
   occurrences survive.
5. Quickselect gives **O(N) average** but O(N²) worst case and it mutates a copy; the size-K
   heap is O(N log K) worst case with O(K) extra space. Prefer quickselect when K is close to
   N, the heap when K ≪ N or when the input is a stream you cannot re-read.
6. Use a **three-way partition** in quickselect. Heavy duplicates — exactly the case the
   interviewer probes — degrade a two-way partition to O(N²).
7. For Shape 2, encode the composite key so that one comparator does everything: primary
   descending, secondary ascending. Sorting the final K by `(-score, distance, index)` keeps
   full ties in input order, which is what "stable" means here.

**Complexity:** size-K heap O(N log K) time / O(K) space; quickselect O(N) average, O(N²)
worst / O(N) space for the copy. Shape 2 is the same, plus O(K log K) to order the result.

**Python Solution:**

```python
import heapq
import random
from dataclasses import dataclass


# ---------- Shape 1: retain the K largest, original order ----------
def retain_k_largest(nums: list[int], k: int) -> list[int]:
    """
    Size-K min-heap. Ties at the cutoff keep the earliest occurrences.
    Time: O(N log K), Space: O(K)
    """
    if k <= 0:
        return []
    if k >= len(nums):
        return list(nums)

    heap: list[tuple[int, int]] = []
    for i, v in enumerate(nums):
        entry = (v, -i)                 # -i: among equal values the LATER index is "smaller"
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)

    keep = {-neg_i for _, neg_i in heap}
    return [v for i, v in enumerate(nums) if i in keep]


def _kth_largest(nums: list[int], k: int) -> int:
    """Iterative quickselect with a random pivot. O(N) average."""
    arr = list(nums)
    target = len(arr) - k               # k-th largest == target-th smallest (0-indexed)
    lo, hi = 0, len(arr) - 1
    while lo < hi:
        pivot = arr[random.randint(lo, hi)]
        i, j, p = lo, hi, lo
        while p <= j:                   # three-way partition handles heavy duplicates
            if arr[p] < pivot:
                arr[i], arr[p] = arr[p], arr[i]
                i += 1
                p += 1
            elif arr[p] > pivot:
                arr[p], arr[j] = arr[j], arr[p]
                j -= 1
            else:
                p += 1
        if target < i:
            hi = i - 1
        elif target > j:
            lo = j + 1
        else:
            return pivot
    return arr[lo]


def retain_k_largest_quickselect(nums: list[int], k: int) -> list[int]:
    """
    Threshold sweep. The tie budget is what keeps the count at exactly K.
    Time: O(N) average, Space: O(N)
    """
    if k <= 0:
        return []
    if k >= len(nums):
        return list(nums)

    threshold = _kth_largest(nums, k)
    strictly_greater = sum(1 for v in nums if v > threshold)
    ties_to_keep = k - strictly_greater    # how many elements equal to the cutoff we may keep

    out = []
    for v in nums:
        if v > threshold:
            out.append(v)
        elif v == threshold and ties_to_keep > 0:
            out.append(v)
            ties_to_keep -= 1
    return out


def retain_k_largest_all_ties(nums: list[int], k: int) -> list[int]:
    """The other defensible rule: keep every element tied at the cutoff (>= K results)."""
    if k <= 0:
        return []
    if k >= len(nums):
        return list(nums)
    threshold = _kth_largest(nums, k)
    return [v for v in nums if v >= threshold]


# ---------- Shape 2: rank stores by a composite key ----------
@dataclass(frozen=True)
class Store:
    name: str
    score: float
    distance: float
    is_open: bool


def top_k_stores(stores: list[Store], k: int, open_only: bool = False) -> list[Store]:
    """
    Top K by score DESC, ties broken by distance ASC, further ties by input order.
    open_only is the clarifying question: does is_open filter, or is it ignored?
    Time: O(N log K + K log K), Space: O(K)
    """
    candidates = [s for s in stores if s.is_open] if open_only else stores
    if k <= 0:
        return []

    # Min-heap keyed so that "smallest" == worst candidate: lowest score,
    # then largest distance, then latest input position.
    heap: list[tuple[float, float, int, Store]] = []
    for i, s in enumerate(candidates):
        entry = (s.score, -s.distance, -i, s)
        if len(heap) < k:
            heapq.heappush(heap, entry)
        elif entry[:3] > heap[0][:3]:
            heapq.heapreplace(heap, entry)

    heap.sort(key=lambda e: (-e[0], -e[1], -e[2]))   # score desc, distance asc, index asc
    return [e[3] for e in heap]


def top_k_stores_oneliner(stores: list[Store], k: int, open_only: bool = False) -> list[Store]:
    """What you write if the interviewer allows the library: nlargest is stable on ties."""
    candidates = [s for s in stores if s.is_open] if open_only else stores
    return heapq.nlargest(k, candidates, key=lambda s: (s.score, -s.distance))


# Example usage
if __name__ == "__main__":
    print(retain_k_largest([3, 1, 5, 2, 4], 3))     # [3, 5, 4]
    print(retain_k_largest([5, 5, 5, 1], 2))        # [5, 5]   (exactly K)
    print(retain_k_largest_all_ties([5, 5, 5, 1], 2))   # [5, 5, 5]   (all ties)

    stores = [
        Store("A", 4.5, 2.0, True), Store("B", 4.9, 5.0, True),
        Store("C", 4.5, 1.0, False), Store("D", 3.0, 0.5, True),
    ]
    print([s.name for s in top_k_stores(stores, 3)])                    # ['B', 'C', 'A']
    print([s.name for s in top_k_stores(stores, 3, open_only=True)])    # ['B', 'A', 'D']
```

## 5. Rate Limiter (Design + Implementation)

| | |
|---|---|
| **Tracks** | SWE · MLE |
| **Tags** | rate-limiting · redis · distributed-systems · throttling · medium |
| **Frequency** | Medium |
| **Last asked** | 2026-04-21 |
| **Stage** | onsite-coding · onsite-system-design |

Recurring Microsoft prompt that crosses categories. Some loops ask for a 30-minute coding
implementation, others for a full system-design treatment, and one HE pairs both in the same
round. The scale follow-up converges on 100K QPS.

**Problem Statement:**

The base ask is `allow(client_id) -> bool` that admits up to N requests per T seconds per
client. Test cases the interviewer drives:

1. Pure rate enforcement (5 requests / 10 seconds).
2. Sliding-window precision — a request that arrived 11 seconds ago should not count.
3. Burst handling — short spikes within the window must be admitted up to N.

Common follow-ups:

- **Scale to 100K QPS.** What pieces sit on the request hot-path; where does state live; what
  is the failure mode.
- **Distributed enforcement.** Multiple gateway nodes; cannot trust local counters.
- **Two rate limits compose** (per-user + global). Reject if either is exceeded.
- **Logger rate-limit variant.** Same shape but the function is
  `shouldPrintMessage(timestamp, message)` and returns true at most once per message per window.

**Test Cases:**

| Scenario | Expectation |
|----------|-------------|
| 5 requests at t=0, limit 5/10s | all admitted (burst up to N is allowed) |
| 6th request at t=0 | rejected |
| request at t=9 | rejected — still inside the window |
| request at t=11 | admitted — the t=0 batch aged out |
| limit 1/10s: admit at t=0, retry at t=9.9 | rejected |
| limit 1/10s: retry at t=10.0 | admitted — the window boundary is inclusive of exactly T |
| two clients, limit 2 | counters are independent |
| token bucket rate 0.5/s burst 5: 5 at t=0, then t=1.9 | rejected — no token has accrued yet |
| same bucket at t=2.0 | admitted — exactly one token accrued |
| same bucket idle until t=1000 | 5 admitted then rejected — refill caps at burst |
| composite (per-user 3, global 4): 4th request from user A | rejected by the per-user limit |
| composite: user B after global is exhausted | rejected, **and B's own bucket is untouched** |
| logger: `(1,"foo")`, `(3,"foo")`, `(11,"foo")` | `true`, `false`, `true` |
| logger: 5000 distinct messages | retained entries stay bounded (eviction works) |

The counter algorithm is an approximation, and the tests pin down **both** directions of its
error against the exact log:

| Burst position | At | Exact log | Sliding-window counter |
|----------------|-----|-----------|------------------------|
| 5 hits at t=0 (window start) | t=11 | admits (they truly aged out) | **falsely rejects** — assumes 0.9 of them still count |
| 5 hits at t=9 (window end) | t=15 | rejects (all 5 are in (5,15]) | **falsely admits** — credits only 0.5 of them |

**Key Insights:**

1. **Sliding-window log** is the textbook starting point — store every request timestamp in a
   deque per client, drop entries outside `(now - T, now]`, admit when `len(deque) < N`.
   Trivially correct, but memory grows with N per active client.
2. **Token bucket is the dominant production choice**: per client, store
   `(current_tokens, last_refill_ts)`. On request, `tokens += (now - last_refill_ts) * rate`,
   cap at burst, decrement on admit. Memory is O(1) per client, refill is implicit (lazy), and
   bursts are naturally supported via the burst cap.
3. Advance `last_refill_ts` even on a **rejected** request. Forgetting this is the common bug:
   the elapsed time gets re-counted on the next call and tokens accrue too fast.
4. **Sliding-window counter** (fixed-window + smooth-by-overlap) is the middle ground when you
   cannot afford the per-client deque but want sub-window precision: keep counters per window,
   and on each request compute the weighted sum of the current and previous window proportional
   to where the rolling window cuts. It is an approximation in both directions (table above) —
   say so out loud rather than claiming it is exact.
5. **Composing two limits requires a check-then-consume split.** If you call
   `per_user.allow(...) and global.allow(...)`, a global rejection has already burned a
   per-user token, and short-circuit evaluation means the two limiters disagree about what was
   admitted. Peek at every limiter first, then consume from all of them.
6. **100K QPS path.** Put state in Redis. Each `allow()` becomes a single Redis call; race
   conditions are eliminated by wrapping the read-modify-write in a **Lua script** so the whole
   token-bucket update is atomic on the Redis side. A single Redis instance saturates around
   the 100K range — shard by `client_id` using consistent hashing across N instances to go
   higher.
7. **Distributed failure modes.** Network partition between gateway and Redis: **fail closed**
   (reject) by default, because admitting unbounded traffic during a Redis outage propagates
   failure downstream. Clock skew across gateway nodes is irrelevant for token-bucket (Redis
   owns the clock) but painful for any algorithm that timestamps on the gateway and merges
   later. Push rule changes to gateways via a config service (ZooKeeper / etcd) rather than
   re-deploying.
8. **Hot-key amplification.** A single very-active `client_id` saturates one Redis shard.
   Mitigations: client-side budget (admit some fraction locally without consulting Redis),
   request batching, and outright blocking abusive sources once detected.
9. For the **logger variant**, the constraint is "at most one print per message per window", so
   it reduces to `last_seen[message]` — admit when `now - last_seen[message] >= 10`. The trick
   is the memory bound: candidates who store every message forever fail the follow-up. The
   correct answer is eviction of entries whose timestamp is older than the window.

**Complexity:** log O(1) amortized per call, O(N) memory per active client. Token bucket and
sliding-window counter are both O(1) time and O(1) memory per client.

**Python Solution:**

```python
import threading
from collections import deque, defaultdict


# ---------- 1. Sliding-window log: the textbook starting point ----------
class SlidingWindowLogLimiter:
    """
    Store every admitted timestamp per client; drop entries outside (now - T, now].
    Exactly correct, but memory is O(N) per active client.
    """

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self.log: dict[str, deque] = defaultdict(deque)
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float) -> bool:
        with self._lock:
            q = self.log[client_id]
            cutoff = now - self.window
            while q and q[0] <= cutoff:       # a request exactly T ago no longer counts
                q.popleft()
            if len(q) < self.limit:
                q.append(now)
                return True
            return False


# ---------- 2. Token bucket: the production choice ----------
class TokenBucketLimiter:
    """
    Per client store (tokens, last_refill). Refill lazily on read:
    tokens += elapsed * rate, capped at burst. O(1) memory per client.
    """

    def __init__(self, rate_per_second: float, burst: int):
        self.rate = rate_per_second
        self.burst = burst
        self.state: dict[str, tuple[float, float]] = {}
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float, cost: float = 1.0) -> bool:
        with self._lock:
            tokens, last = self.state.get(client_id, (float(self.burst), now))
            tokens = min(self.burst, tokens + (now - last) * self.rate)   # lazy refill
            if tokens >= cost:
                self.state[client_id] = (tokens - cost, now)
                return True
            self.state[client_id] = (tokens, now)      # still advance the clock
            return False


# ---------- 3. Sliding-window counter: the middle ground ----------
class SlidingWindowCounterLimiter:
    """
    Two fixed-window counters, blended by how far into the current window we are.
    O(1) memory per client, no per-request deque, sub-window precision --
    but an APPROXIMATION: it assumes the previous window's hits were uniform.
    """

    def __init__(self, limit: int, window_seconds: float):
        self.limit = limit
        self.window = window_seconds
        self.state: dict[str, tuple[int, int, int]] = {}   # client -> (window_id, cur, prev)
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float) -> bool:
        with self._lock:
            wid = int(now // self.window)
            prev_wid, cur, prev = self.state.get(client_id, (wid, 0, 0))
            if wid == prev_wid + 1:
                prev, cur = cur, 0
            elif wid > prev_wid:
                prev, cur = 0, 0

            overlap = 1.0 - (now % self.window) / self.window   # share of the previous window
            estimate = prev * overlap + cur
            if estimate + 1 <= self.limit:
                self.state[client_id] = (wid, cur + 1, prev)
                return True
            self.state[client_id] = (wid, cur, prev)
            return False


# ---------- 4. Composite limits: reject if EITHER is exceeded ----------
class CompositeLimiter:
    """
    Per-user plus global. Check every limiter BEFORE consuming from any of them,
    otherwise a rejection by the second limiter still burns a token in the first.
    """

    def __init__(self, per_user: TokenBucketLimiter, global_limiter: TokenBucketLimiter):
        self.per_user = per_user
        self.global_limiter = global_limiter
        self._lock = threading.Lock()

    def allow(self, client_id: str, now: float) -> bool:
        with self._lock:
            if not self._peek(self.per_user, client_id, now):
                return False
            if not self._peek(self.global_limiter, "__global__", now):
                return False
            self.per_user.allow(client_id, now)
            self.global_limiter.allow("__global__", now)
            return True

    @staticmethod
    def _peek(bucket: TokenBucketLimiter, key: str, now: float) -> bool:
        """Would this admit, without consuming?"""
        tokens, last = bucket.state.get(key, (float(bucket.burst), now))
        return min(bucket.burst, tokens + (now - last) * bucket.rate) >= 1.0


# ---------- 5. Logger variant (LC 359) ----------
class Logger:
    """
    Print a message at most once per window. last_seen alone leaks memory on an
    unbounded message alphabet, so evict entries older than the window.
    """

    def __init__(self, window_seconds: int = 10, evict_every: int = 1000):
        self.window = window_seconds
        self.last_seen: dict[str, int] = {}
        self.evict_every = evict_every
        self._since_evict = 0

    def should_print_message(self, timestamp: int, message: str) -> bool:
        self._since_evict += 1
        if self._since_evict >= self.evict_every:
            self._evict(timestamp)
            self._since_evict = 0

        last = self.last_seen.get(message)
        if last is None or timestamp - last >= self.window:
            self.last_seen[message] = timestamp
            return True
        return False

    def _evict(self, now: int) -> None:
        stale = [m for m, ts in self.last_seen.items() if now - ts >= self.window]
        for m in stale:
            del self.last_seen[m]


# Example usage
if __name__ == "__main__":
    rl = SlidingWindowLogLimiter(5, 10)          # 5 requests / 10 seconds
    print([rl.allow("u", 0.0) for _ in range(6)])
    # [True, True, True, True, True, False]
    print(rl.allow("u", 11.0))                   # True  (the t=0 batch aged out)

    tb = TokenBucketLimiter(rate_per_second=0.5, burst=5)
    print([tb.allow("u", 0.0) for _ in range(5)])   # burst admitted
    print(tb.allow("u", 1.9), tb.allow("u", 2.0))   # False True  (one token accrues at t=2)

    lg = Logger(10)
    print(lg.should_print_message(1, "foo"),
          lg.should_print_message(3, "foo"),
          lg.should_print_message(11, "foo"))       # True False True
```

## 6. Job Scheduler / ETL Pipeline System Design

| | |
|---|---|
| **Tracks** | SWE |
| **Tags** | distributed-systems · scheduling · etl · messaging · medium |
| **Frequency** | Single report |
| **Last asked** | 2026-01-26 |
| **Stage** | onsite-system-design |

HE SD slot. Design a system that schedules and executes long-running ETL jobs reliably, with
retry and dependency semantics.

**Functional Requirements:**

- Register a job with a schedule (cron-like) and a dependency on other jobs.
- Execute on time, exactly once per scheduled tick (or at-least-once with downstream
  idempotency).
- Track job status (pending / running / succeeded / failed / retrying).
- Surface a job-level retry policy.
- Provide an admin UI / API for inspection and manual triggers.

**Non-Functional Requirements:**

- 10K active job definitions.
- Per-tick fan-out can spike to thousands of concurrent runs.
- A run may last seconds to many hours.
- Survive worker crashes mid-run.

**High-Level Design:**

```
                  ┌──────────────┐
   admin UI/API ──│ LB + cache   │
                  └──────┬───────┘
                         │
   ┌─────────────────────▼──────────────────────┐
   │  Scheduler (single leader, ZK/etcd lease)  │
   │  • wakes per minute, finds due jobs        │
   │  • enqueues run tasks                      │
   │  • reaps expired worker leases             │
   └──────┬──────────────────────────┬──────────┘
          │ enqueue                  │ read/write
   ┌──────▼───────────┐      ┌───────▼─────────────────────┐
   │ Durable queue    │      │ State store (PostgreSQL)    │
   │ (Kafka / SQS)    │      │ jobs, runs, attempts, DAG   │
   │ (run_id,job_id,  │      └───────▲─────────────────────┘
   │  attempt)        │              │
   └──────┬───────────┘              │ status + heartbeat
          │ pull                     │
   ┌──────▼──────────────────────────┴──────────┐
   │  Stateless worker pool (autoscale on depth)│
   └────────────────────────────────────────────┘
```

1. **Scheduler** — stateful service holding the schedule store. Wakes per minute, computes
   which jobs are due, enqueues run tasks. **Single-leader** (with standby failover via
   ZooKeeper / etcd) to avoid double-scheduling.
2. **Job queue** — durable queue (Kafka / SQS) holding `(run_id, job_id, attempt)`. Workers pull.
3. **Worker pool** — stateless executors, autoscaled on queue depth. Each worker picks a task,
   marks the run `running` in the durable store, executes, and marks `succeeded` / `failed`.
4. **State store** — durable DB (PostgreSQL) recording job definitions, runs, retry counts, and
   the dependency graph.
5. **Dependency engine** — when a run completes, look up downstream jobs in the DAG; enqueue
   them if their other upstreams are also satisfied.
6. **Load balancer + cache** for the admin UI / API.

**Exactly-once semantics:** true exactly-once is hard; the practical answer is **at-least-once
enqueue + idempotent job logic**, enforced via run-id-keyed dedup at the worker boundary. The
worker checks `state_store.run_status(run_id)`; if the run is already `running` or `succeeded`
under another worker, it skips. Make `run_id` deterministic — `hash(job_id, scheduled_tick)` —
so a redelivered or re-fired tick maps to the same row instead of creating a second run.

**Worker-crash recovery:** the worker holds a **heartbeat lease** on its run (a TTL on the
state-store row). On crash the lease expires, the scheduler reaps it and re-enqueues with
`attempt += 1` up to the retry policy max. The reaped worker must not be able to commit
afterwards — check lease ownership on completion, otherwise a worker that merely stalled (GC
pause, network blip) will report success for a run that has already been re-executed.

**Retry policy:** exponential backoff with **jitter**, capped at max-attempts. Full jitter
matters at this fan-out: thousands of runs failing on the same downstream outage will otherwise
retry in lockstep and knock it over again. Permanent failures move to a **dead-letter** state
for operator review.

**Dependency graph:** cycle detection at registration time (Kahn topological sort) — reject the
registration rather than discovering the cycle at 3am. At run time, when a node finishes, only
its **direct downstreams** are evaluated; keep the engine local rather than re-computing the
full DAG. A downstream fires only when *every* upstream succeeded for that same tick.

**Data Model (sketch):**

```
jobs(job_id PK, cron_expr, max_attempts, timeout_s, enabled, created_at)
job_deps(upstream_id, downstream_id, PK(upstream_id, downstream_id))
runs(run_id PK,             -- deterministic: hash(job_id, scheduled_tick)
     job_id, scheduled_tick, attempt, status,
     worker_id, lease_expires_at, started_at, finished_at, error)
  INDEX (status, lease_expires_at)   -- the reaper's scan
  INDEX (job_id, scheduled_tick)     -- dependency lookups
```

**Observability:** per-job latency histogram, success rate, and queue-depth dashboards. Alert
on rising queue depth (workers under-provisioned) or rising failure rate.

**Reference implementation of the algorithmic core** (the parts an interviewer may ask you to
actually write — cycle detection, downstream readiness, lease reaping, retry/backoff):

```python
import random
from collections import defaultdict, deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class RunStatus(str, Enum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    RETRYING = "retrying"
    DEAD_LETTER = "dead_letter"


@dataclass
class Job:
    job_id: str
    upstreams: frozenset = frozenset()
    max_attempts: int = 3


@dataclass
class Run:
    run_id: str
    job_id: str
    scheduled_tick: int
    attempt: int = 1
    status: RunStatus = RunStatus.PENDING
    lease_expires_at: float = 0.0
    worker_id: Optional[str] = None


class CycleError(ValueError):
    pass


# ---------- Registration: reject cycles up front (Kahn topological sort) ----------
class JobRegistry:
    def __init__(self):
        self.jobs: dict[str, Job] = {}
        self.downstreams: dict[str, set[str]] = defaultdict(set)

    def register(self, job: Job) -> None:
        """Registering a job that would close a cycle must fail, not corrupt the graph."""
        if job.job_id in self.jobs:
            raise ValueError(f"job {job.job_id!r} already registered")
        missing = job.upstreams - self.jobs.keys()
        if missing:
            raise ValueError(f"unknown upstream(s): {sorted(missing)}")

        self.jobs[job.job_id] = job
        for up in job.upstreams:
            self.downstreams[up].add(job.job_id)

        if not self._is_acyclic():
            for up in job.upstreams:          # roll back so the registry stays usable
                self.downstreams[up].discard(job.job_id)
            del self.jobs[job.job_id]
            raise CycleError(f"registering {job.job_id!r} would create a cycle")

    def register_edge(self, upstream: str, downstream: str) -> None:
        """Add a dependency between two already-registered jobs."""
        for j in (upstream, downstream):
            if j not in self.jobs:
                raise ValueError(f"unknown job {j!r}")
        job = self.jobs[downstream]
        self.jobs[downstream] = Job(job.job_id, job.upstreams | {upstream}, job.max_attempts)
        self.downstreams[upstream].add(downstream)
        if not self._is_acyclic():
            self.jobs[downstream] = job
            self.downstreams[upstream].discard(downstream)
            raise CycleError(f"edge {upstream!r} -> {downstream!r} would create a cycle")

    def _is_acyclic(self) -> bool:
        """Kahn: if a topological order covers every node, there is no cycle."""
        indegree = {j: len(self.jobs[j].upstreams) for j in self.jobs}
        queue = deque(j for j, d in indegree.items() if d == 0)
        seen = 0
        while queue:
            node = queue.popleft()
            seen += 1
            for down in self.downstreams.get(node, ()):
                indegree[down] -= 1
                if indegree[down] == 0:
                    queue.append(down)
        return seen == len(self.jobs)

    def topological_order(self) -> list[str]:
        indegree = {j: len(self.jobs[j].upstreams) for j in self.jobs}
        queue = deque(sorted(j for j, d in indegree.items() if d == 0))
        order = []
        while queue:
            node = queue.popleft()
            order.append(node)
            for down in sorted(self.downstreams.get(node, ())):
                indegree[down] -= 1
                if indegree[down] == 0:
                    queue.append(down)
        if len(order) != len(self.jobs):
            raise CycleError("graph contains a cycle")
        return order


# ---------- Dependency engine + at-least-once execution ----------
class Orchestrator:
    """
    In-memory stand-in for (scheduler + durable state store + queue). The point is
    the semantics: local downstream evaluation, run-id dedup, lease reaping, retries.
    """

    def __init__(self, registry: JobRegistry, lease_seconds: float = 30.0,
                 base_backoff: float = 1.0, max_backoff: float = 60.0):
        self.registry = registry
        self.lease_seconds = lease_seconds
        self.base_backoff = base_backoff
        self.max_backoff = max_backoff
        self.runs: dict[str, Run] = {}
        self.queue: deque = deque()
        self.dead_letter: list[str] = []

    @staticmethod
    def run_id(job_id: str, tick: int) -> str:
        """Deterministic per (job, tick): the dedup key that makes re-enqueue safe."""
        return f"{job_id}@{tick}"

    def enqueue_due(self, due_job_ids: list[str], tick: int) -> list[str]:
        """Scheduler tick. Only roots are scheduled; downstreams trigger on completion."""
        enqueued = []
        for job_id in due_job_ids:
            rid = self.run_id(job_id, tick)
            if rid in self.runs:              # idempotent: a re-fired tick must not duplicate
                continue
            self.runs[rid] = Run(rid, job_id, tick)
            self.queue.append(rid)
            enqueued.append(rid)
        return enqueued

    def claim(self, worker_id: str, now: float) -> Optional[Run]:
        """
        Worker pulls a task. At-least-once delivery means the same run_id can arrive
        twice; the status check at the worker boundary is what makes that safe.
        """
        while self.queue:
            rid = self.queue.popleft()
            run = self.runs[rid]
            if run.status in (RunStatus.RUNNING, RunStatus.SUCCEEDED, RunStatus.DEAD_LETTER):
                continue                      # already claimed or finished: drop the duplicate
            run.status = RunStatus.RUNNING
            run.worker_id = worker_id
            run.lease_expires_at = now + self.lease_seconds
            return run
        return None

    def heartbeat(self, run_id: str, worker_id: str, now: float) -> bool:
        """Extend the lease. A worker that lost its lease must not keep going."""
        run = self.runs[run_id]
        if run.worker_id != worker_id or run.status is not RunStatus.RUNNING:
            return False
        run.lease_expires_at = now + self.lease_seconds
        return True

    def complete(self, run_id: str, worker_id: str, tick: int) -> list[str]:
        """Mark succeeded and enqueue any downstream whose other upstreams are satisfied."""
        run = self.runs[run_id]
        if run.worker_id != worker_id or run.status is not RunStatus.RUNNING:
            return []                         # a reaped worker's late completion is ignored
        run.status = RunStatus.SUCCEEDED
        return self._enqueue_ready_downstreams(run.job_id, tick)

    def fail(self, run_id: str, worker_id: str, now: float) -> RunStatus:
        """Retry with exponential backoff + jitter, or dead-letter at max attempts."""
        run = self.runs[run_id]
        if run.worker_id != worker_id or run.status is not RunStatus.RUNNING:
            return run.status
        return self._retry_or_dead_letter(run, now)

    def reap_expired_leases(self, now: float) -> list[str]:
        """Scheduler-side crash recovery: an expired lease means the worker died."""
        reaped = []
        for run in list(self.runs.values()):
            if run.status is RunStatus.RUNNING and run.lease_expires_at <= now:
                self._retry_or_dead_letter(run, now)
                reaped.append(run.run_id)
        return reaped

    def _retry_or_dead_letter(self, run: Run, now: float) -> RunStatus:
        job = self.registry.jobs[run.job_id]
        run.worker_id = None                  # revoke the lease: the old worker cannot commit
        if run.attempt >= job.max_attempts:
            run.status = RunStatus.DEAD_LETTER
            self.dead_letter.append(run.run_id)
            return RunStatus.DEAD_LETTER
        run.attempt += 1
        run.status = RunStatus.RETRYING
        self.queue.append(run.run_id)
        return RunStatus.RETRYING

    def backoff_seconds(self, attempt: int) -> float:
        """Exponential backoff, capped, with full jitter to avoid a retry thundering herd."""
        capped = min(self.max_backoff, self.base_backoff * (2 ** (attempt - 1)))
        return random.uniform(0, capped)

    def _enqueue_ready_downstreams(self, finished_job_id: str, tick: int) -> list[str]:
        """
        Only direct downstreams are evaluated — never re-walk the whole DAG.
        A downstream fires when EVERY upstream succeeded for this same tick.
        """
        enqueued = []
        for down in sorted(self.registry.downstreams.get(finished_job_id, ())):
            rid = self.run_id(down, tick)
            if rid in self.runs:
                continue
            upstreams = self.registry.jobs[down].upstreams
            if all(self.runs.get(self.run_id(u, tick), Run("", "", 0)).status
                   is RunStatus.SUCCEEDED for u in upstreams):
                self.runs[rid] = Run(rid, down, tick)
                self.queue.append(rid)
                enqueued.append(rid)
        return enqueued


# Example usage
if __name__ == "__main__":
    reg = JobRegistry()
    reg.register(Job("extract"))
    reg.register(Job("transform", frozenset({"extract"})))
    reg.register(Job("load", frozenset({"transform"})))
    print(reg.topological_order())          # ['extract', 'transform', 'load']

    try:
        reg.register_edge("load", "extract")
    except CycleError as e:
        print("rejected:", e)               # rejected: edge 'load' -> 'extract' ...

    orc = Orchestrator(reg, lease_seconds=30.0)
    print(orc.enqueue_due(["extract"], tick=100))     # ['extract@100']

    run = orc.claim("worker-1", now=0.0)
    print(orc.complete(run.run_id, "worker-1", 100))  # ['transform@100']

    # worker-2 claims the next run, then crashes: the lease expires and it is retried
    orc.claim("worker-2", now=1.0)
    print(orc.reap_expired_leases(now=31.0))          # ['transform@100']
    print(orc.runs["transform@100"].status,
          orc.runs["transform@100"].attempt)          # RunStatus.RETRYING 2
```

**Behavior pinned down by the test suite:**

| Scenario | Expectation |
|----------|-------------|
| register a back-edge (`load → extract`) | `CycleError`, and the registry is left unchanged |
| register with an unknown upstream | `ValueError` |
| diamond `a → {b,c} → d` | topological order `[a, b, c, d]` |
| same tick fired twice | second call enqueues nothing |
| queue redelivers the same `run_id` | only one worker gets a run; the duplicate is dropped |
| fan-in: `b` finishes, `c` has not | `d` is **not** enqueued |
| fan-in: `c` then finishes | `d` is enqueued exactly once |
| lease expires at exactly T | reaped; `attempt` becomes 2, status `retrying` |
| reaped worker reports success afterwards | ignored; the run stays `retrying` |
| reaped worker sends a heartbeat | refused |
| 3 failures with `max_attempts=3` | `dead_letter`, and the run is not reclaimable |
| upstream dead-letters | downstream is never enqueued |
| backoff for attempts 1-11 | always within `[0, min(max_backoff, base·2^(n-1))]`, and jittered |
| two different ticks of the same job | independent runs; tick 1's completion only triggers tick 1's downstream |

**Follow-up Questions:**

1. **Long runs vs lease TTL** — a 6-hour job with a 30-second lease needs continuous
   heartbeating; what happens if the heartbeat path is slower than the TTL under load?
2. **Backfill** — an operator wants to re-run last month's ticks. How do you bound the fan-out
   so a backfill does not starve live traffic?
3. **Scheduler failover** — the leader dies mid-tick after enqueueing half the due jobs. What
   does the standby do on takeover? (This is why `run_id` is deterministic.)
4. **Skew** — one job's runs take 100× longer than everything else and monopolize the pool.
   Per-job concurrency caps, or separate queues?
5. **Cron ambiguity** — DST transitions and missed ticks while the scheduler was down: skip,
   run once, or catch up all of them?

**Preparation:**

- Pre-write the four-component diagram (scheduler / queue / workers / state store).
- Know the exactly-once → at-least-once + idempotency framing; this is the standard answer.
- Drill the worker-crash story: heartbeat lease, scheduler reap, re-enqueue with attempt
  counter — and mention revoking the old worker's ability to commit.
- Pre-rehearse the dependency DAG handling and the cycle-detection step at registration.

---

## References

- LeetCode 146 — LRU Cache; LeetCode 460 — LFU Cache; LeetCode 359 — Logger Rate Limiter
- LeetCode 215 — Kth Largest Element (the selection core of the top-K problem)
- Hierholzer's algorithm for Eulerian paths (Part 2 of the shotgun sequencing problem)
- RFC 4180 — the CSV conventions the in-memory DB parser has to honor
