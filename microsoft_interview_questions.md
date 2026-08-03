# Microsoft Interview Questions

Coding problems reported from Microsoft loops, with emphasis on the MAI (Microsoft AI) /
Copilot platform rounds. Every solution below is implemented and tested in Python, Java, and
TypeScript.

---

## Table of Contents

1. [DNA Shotgun Sequencing](#1-dna-shotgun-sequencing)
2. [In-Memory SQL Engine](#2-in-memory-sql-engine)
3. [LRU Cache (LC 146) + Multithreading Variant](#3-lru-cache-lc-146--multithreading-variant)

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

**Java Solution:**

```java
import java.util.*;
import java.util.stream.Collectors;

public class ShotgunSequencing {

    public record Sequence(String startId, String endId, String payload) {}

    // ---------- Part 1: directed chain ----------
    // Time: O(N), Space: O(N)
    public static String shotgunSequence(List<Sequence> sequences) {
        if (sequences.isEmpty()) return "";

        Map<String, Sequence> byStart = new HashMap<>();
        Set<String> ends = new HashSet<>();
        for (Sequence s : sequences) {
            if (byStart.put(s.startId(), s) != null) {
                throw new IllegalArgumentException("duplicate start tag " + s.startId());
            }
            ends.add(s.endId());
        }

        // The head is the only start tag that is nobody's end tag.
        String head = null;
        int heads = 0;
        for (Sequence s : sequences) {
            if (!ends.contains(s.startId())) { head = s.startId(); heads++; }
        }
        if (heads != 1) {
            throw new IllegalArgumentException("expected exactly one head, found " + heads);
        }

        StringBuilder sb = new StringBuilder();
        String tag = head;
        int steps = 0;
        while (byStart.containsKey(tag) && steps < sequences.size()) {
            Sequence s = byStart.get(tag);
            sb.append(s.payload());
            tag = s.endId();
            steps++;
        }
        if (steps != sequences.size()) {
            throw new IllegalArgumentException("fragments do not form a single chain");
        }
        return sb.toString();
    }

    // ---------- Part 2: undirected chain (Eulerian path) ----------
    private record Edge(String to, int id) {}

    // Iterative Hierholzer. Time: O(V + E), Space: O(V + E)
    public static String shotgunSequenceUndirected(List<Sequence> sequences) {
        int m = sequences.size();
        if (m == 0) return "";

        // Each fragment is one undirected edge keyed by index, so duplicate
        // fragments stay distinct parallel edges.
        Map<String, List<Edge>> adj = new HashMap<>();
        Map<String, Integer> degree = new HashMap<>();
        List<String> order = new ArrayList<>();     // first-seen order, deterministic start
        for (int i = 0; i < m; i++) {
            Sequence s = sequences.get(i);
            for (String t : List.of(s.startId(), s.endId())) {
                if (!degree.containsKey(t)) order.add(t);
                degree.merge(t, 1, Integer::sum);
            }
            adj.computeIfAbsent(s.startId(), k -> new ArrayList<>()).add(new Edge(s.endId(), i));
            if (!s.startId().equals(s.endId())) {   // self-loop stored once, degree counted twice
                adj.computeIfAbsent(s.endId(), k -> new ArrayList<>()).add(new Edge(s.startId(), i));
            }
        }

        List<String> odd = order.stream().filter(t -> degree.get(t) % 2 == 1)
                                .collect(Collectors.toList());
        if (odd.size() != 0 && odd.size() != 2) {
            throw new IllegalArgumentException(
                    "no Eulerian path: " + odd.size() + " odd-degree tags");
        }
        String start = odd.isEmpty() ? order.get(0) : odd.get(0);

        Map<String, Integer> ptr = new HashMap<>();  // per-vertex cursor keeps this O(E)
        boolean[] used = new boolean[m];
        Deque<String> stackV = new ArrayDeque<>();   // (vertex, edge used to arrive)
        Deque<Integer> stackE = new ArrayDeque<>();
        stackV.push(start);
        stackE.push(-1);
        List<String> pathV = new ArrayList<>();
        List<Integer> pathE = new ArrayList<>();

        while (!stackV.isEmpty()) {
            String v = stackV.peek();
            List<Edge> lst = adj.getOrDefault(v, List.of());
            int p = ptr.getOrDefault(v, 0);
            while (p < lst.size() && used[lst.get(p).id()]) p++;
            if (p == lst.size()) {
                ptr.put(v, p);
                pathV.add(stackV.pop());             // vertex exhausted: splice into path
                pathE.add(stackE.pop());
            } else {
                Edge e = lst.get(p);
                ptr.put(v, p + 1);
                used[e.id()] = true;
                stackV.push(e.to());
                stackE.push(e.id());
            }
        }
        Collections.reverse(pathV);
        Collections.reverse(pathE);
        if (pathV.size() != m + 1) {
            throw new IllegalArgumentException(
                    "disconnected: no single traversal uses every fragment");
        }

        // Reverse the payload when the edge is walked against its declared direction.
        StringBuilder sb = new StringBuilder();
        for (int i = 1; i < pathV.size(); i++) {
            Sequence s = sequences.get(pathE.get(i));
            String prev = pathV.get(i - 1);
            sb.append(s.startId().equals(prev)
                    ? s.payload()
                    : new StringBuilder(s.payload()).reverse());
        }
        return sb.toString();
    }

    // ---------- Part 3: multi-chain decomposition ----------
    public record Decomposition(List<String> chains, List<Sequence> ambiguous) {}

    // Time: O(N * alpha(N)), Space: O(N)
    public static Decomposition decomposeChains(List<Sequence> sequences) {
        Map<String, String> parent = new HashMap<>();
        for (Sequence s : sequences) union(parent, s.startId(), s.endId());

        Map<String, List<Integer>> groups = new LinkedHashMap<>();
        for (int i = 0; i < sequences.size(); i++) {
            groups.computeIfAbsent(find(parent, sequences.get(i).startId()),
                                   k -> new ArrayList<>()).add(i);
        }

        List<int[]> chainHeads = new ArrayList<>();   // {headFragmentIndex, slot}
        List<String> assembled = new ArrayList<>();
        List<Integer> ambiguous = new ArrayList<>();

        for (List<Integer> idxs : groups.values()) {
            Map<String, Integer> byStart = new HashMap<>();
            Map<String, Integer> endCount = new HashMap<>();
            boolean conflict = false;
            for (int i : idxs) {
                Sequence s = sequences.get(i);
                if (byStart.put(s.startId(), i) != null) conflict = true;
                endCount.merge(s.endId(), 1, Integer::sum);
            }
            if (conflict || endCount.values().stream().anyMatch(c -> c > 1)) {
                ambiguous.addAll(idxs);
                continue;
            }
            List<String> heads = idxs.stream()
                    .map(i -> sequences.get(i).startId())
                    .filter(t -> !endCount.containsKey(t))
                    .collect(Collectors.toList());
            if (heads.size() != 1) { ambiguous.addAll(idxs); continue; }   // 0 heads == cycle

            int headIdx = byStart.get(heads.get(0));
            StringBuilder sb = new StringBuilder();
            String tag = heads.get(0);
            int steps = 0;
            while (byStart.containsKey(tag) && steps < idxs.size()) {
                Sequence s = sequences.get(byStart.get(tag));
                sb.append(s.payload());
                tag = s.endId();
                steps++;
            }
            if (steps != idxs.size()) { ambiguous.addAll(idxs); continue; }
            chainHeads.add(new int[]{headIdx, assembled.size()});
            assembled.add(sb.toString());
        }

        chainHeads.sort(Comparator.comparingInt(a -> a[0]));
        List<String> chains = chainHeads.stream().map(a -> assembled.get(a[1]))
                                        .collect(Collectors.toList());
        Collections.sort(ambiguous);
        List<Sequence> ambiguousSeqs = ambiguous.stream().map(sequences::get)
                                                .collect(Collectors.toList());
        return new Decomposition(chains, ambiguousSeqs);
    }

    public static List<String> shotgunSequenceMulti(List<Sequence> sequences) {
        Decomposition d = decomposeChains(sequences);
        if (!d.ambiguous().isEmpty()) {
            throw new IllegalArgumentException(
                    d.ambiguous().size() + " fragment(s) do not form a clean chain");
        }
        return d.chains();
    }

    private static String find(Map<String, String> parent, String x) {
        parent.putIfAbsent(x, x);
        while (!parent.get(x).equals(x)) {
            parent.put(x, parent.get(parent.get(x)));   // path halving
            x = parent.get(x);
        }
        return x;
    }

    private static void union(Map<String, String> parent, String a, String b) {
        String ra = find(parent, a), rb = find(parent, b);
        if (!ra.equals(rb)) parent.put(ra, rb);
    }

    public static void main(String[] args) {
        System.out.println(shotgunSequence(List.of(
                new Sequence("AAA", "AAC", "AAAA"), new Sequence("AGG", "ACC", "GGGG"),
                new Sequence("AAC", "ACT", "TTTT"), new Sequence("ACT", "AGG", "CCCC"))));
        // AAAATTTTCCCCGGGG

        System.out.println(shotgunSequenceUndirected(List.of(
                new Sequence("A", "B", "AAAA"), new Sequence("B", "C", "TTTT"),
                new Sequence("C", "D", "CCCC"), new Sequence("D", "B", "GGGG"))));
        // AAAATTTTCCCCGGGG

        System.out.println(shotgunSequenceMulti(List.of(
                new Sequence("AAA", "AAC", "AAAA"), new Sequence("TGG", "TGA", "CCCC"),
                new Sequence("AAC", "ACT", "TTTT"), new Sequence("TGA", "TAA", "GGGG"))));
        // [AAAATTTT, CCCCGGGG]
    }
}
```

**TypeScript Solution:**

```typescript
interface Sequence {
    startId: string;
    endId: string;
    payload: string;
}

// ---------- Part 1: directed chain ----------
// Time: O(N), Space: O(N)
function shotgunSequence(sequences: Sequence[]): string {
    if (sequences.length === 0) return "";

    const byStart = new Map<string, Sequence>();
    const ends = new Set<string>();
    for (const s of sequences) {
        if (byStart.has(s.startId)) throw new Error(`duplicate start tag ${s.startId}`);
        byStart.set(s.startId, s);
        ends.add(s.endId);
    }

    // The head is the only start tag that is nobody's end tag.
    const heads = sequences.filter((s) => !ends.has(s.startId));
    if (heads.length !== 1) throw new Error(`expected exactly one head, found ${heads.length}`);

    const parts: string[] = [];
    let tag = heads[0].startId;
    while (byStart.has(tag) && parts.length < sequences.length) {
        const s = byStart.get(tag)!;
        parts.push(s.payload);
        tag = s.endId;
    }
    if (parts.length !== sequences.length) throw new Error("fragments do not form a single chain");
    return parts.join("");
}

// ---------- Part 2: undirected chain (Eulerian path) ----------
// Iterative Hierholzer. Time: O(V + E), Space: O(V + E)
function shotgunSequenceUndirected(sequences: Sequence[]): string {
    const m = sequences.length;
    if (m === 0) return "";

    // Each fragment is one undirected edge keyed by index, so duplicate
    // fragments stay distinct parallel edges.
    const adj = new Map<string, Array<[string, number]>>();
    const degree = new Map<string, number>();
    const order: string[] = [];                 // first-seen order, deterministic start
    const addEdge = (from: string, to: string, id: number) => {
        if (!adj.has(from)) adj.set(from, []);
        adj.get(from)!.push([to, id]);
    };
    sequences.forEach((s, i) => {
        for (const t of [s.startId, s.endId]) {
            if (!degree.has(t)) order.push(t);
            degree.set(t, (degree.get(t) ?? 0) + 1);
        }
        addEdge(s.startId, s.endId, i);
        if (s.startId !== s.endId) addEdge(s.endId, s.startId, i);   // self-loop stored once
    });

    const odd = order.filter((t) => degree.get(t)! % 2 === 1);
    if (odd.length !== 0 && odd.length !== 2) {
        throw new Error(`no Eulerian path: ${odd.length} odd-degree tags`);
    }
    const start = odd.length ? odd[0] : order[0];

    const ptr = new Map<string, number>();       // per-vertex cursor keeps this O(E)
    const used = new Array<boolean>(m).fill(false);
    const stack: Array<[string, number]> = [[start, -1]];   // (vertex, arriving edge)
    const path: Array<[string, number]> = [];

    while (stack.length) {
        const [v] = stack[stack.length - 1];
        const lst = adj.get(v) ?? [];
        let p = ptr.get(v) ?? 0;
        while (p < lst.length && used[lst[p][1]]) p++;
        if (p === lst.length) {
            ptr.set(v, p);
            path.push(stack.pop()!);             // vertex exhausted: splice into path
        } else {
            const [u, eid] = lst[p];
            ptr.set(v, p + 1);
            used[eid] = true;
            stack.push([u, eid]);
        }
    }
    path.reverse();
    if (path.length !== m + 1) {
        throw new Error("disconnected: no single traversal uses every fragment");
    }

    // Reverse the payload when the edge is walked against its declared direction.
    const out: string[] = [];
    for (let i = 1; i < path.length; i++) {
        const prev = path[i - 1][0];
        const s = sequences[path[i][1]];
        out.push(s.startId === prev ? s.payload : [...s.payload].reverse().join(""));
    }
    return out.join("");
}

// ---------- Part 3: multi-chain decomposition ----------
interface Decomposition {
    chains: string[];
    ambiguous: Sequence[];
}

// Time: O(N * alpha(N)), Space: O(N)
function decomposeChains(sequences: Sequence[]): Decomposition {
    const parent = new Map<string, string>();
    const find = (x: string): string => {
        if (!parent.has(x)) parent.set(x, x);
        while (parent.get(x) !== x) {
            parent.set(x, parent.get(parent.get(x)!)!);   // path halving
            x = parent.get(x)!;
        }
        return x;
    };
    const union = (a: string, b: string) => {
        const ra = find(a), rb = find(b);
        if (ra !== rb) parent.set(ra, rb);
    };
    for (const s of sequences) union(s.startId, s.endId);

    const groups = new Map<string, number[]>();
    sequences.forEach((s, i) => {
        const root = find(s.startId);
        if (!groups.has(root)) groups.set(root, []);
        groups.get(root)!.push(i);
    });

    const chains: Array<[number, string]> = [];
    const ambiguous: number[] = [];

    for (const idxs of groups.values()) {
        const byStart = new Map<string, number>();
        const endCount = new Map<string, number>();
        let conflict = false;
        for (const i of idxs) {
            const s = sequences[i];
            if (byStart.has(s.startId)) conflict = true;   // a tag starting two fragments branches
            byStart.set(s.startId, i);
            endCount.set(s.endId, (endCount.get(s.endId) ?? 0) + 1);
        }
        if (conflict || [...endCount.values()].some((c) => c > 1)) {
            ambiguous.push(...idxs);
            continue;
        }
        const heads = idxs.map((i) => sequences[i].startId).filter((t) => !endCount.has(t));
        if (heads.length !== 1) { ambiguous.push(...idxs); continue; }   // 0 heads == cycle

        const headIdx = byStart.get(heads[0])!;
        const parts: string[] = [];
        let tag = heads[0];
        while (byStart.has(tag) && parts.length < idxs.length) {
            const s = sequences[byStart.get(tag)!];
            parts.push(s.payload);
            tag = s.endId;
        }
        if (parts.length !== idxs.length) { ambiguous.push(...idxs); continue; }
        chains.push([headIdx, parts.join("")]);
    }

    chains.sort((a, b) => a[0] - b[0]);      // report in input order of each chain's head
    ambiguous.sort((a, b) => a - b);
    return { chains: chains.map(([, c]) => c), ambiguous: ambiguous.map((i) => sequences[i]) };
}

function shotgunSequenceMulti(sequences: Sequence[]): string[] {
    const { chains, ambiguous } = decomposeChains(sequences);
    if (ambiguous.length) {
        throw new Error(`${ambiguous.length} fragment(s) do not form a clean chain`);
    }
    return chains;
}

// Example usage
const seq = (startId: string, endId: string, payload: string): Sequence =>
    ({ startId, endId, payload });

console.log(shotgunSequence([
    seq("AAA", "AAC", "AAAA"), seq("AGG", "ACC", "GGGG"),
    seq("AAC", "ACT", "TTTT"), seq("ACT", "AGG", "CCCC")]));
// AAAATTTTCCCCGGGG

console.log(shotgunSequenceUndirected([
    seq("A", "B", "AAAA"), seq("B", "C", "TTTT"),
    seq("C", "D", "CCCC"), seq("D", "B", "GGGG")]));
// AAAATTTTCCCCGGGG

console.log(shotgunSequenceMulti([
    seq("AAA", "AAC", "AAAA"), seq("TGG", "TGA", "CCCC"),
    seq("AAC", "ACT", "TTTT"), seq("TGA", "TAA", "GGGG")]));
// [ 'AAAATTTT', 'CCCCGGGG' ]
```

**Common Failure Modes (reported):**

- Treating Part 2 as another directed walk and missing the reversal of `payload` on the
  return leg.
- Using DFS recursion for Hierholzer on long chains and blowing the call stack — write the
  iterative version. (The implementations above were exercised on a 200,000-fragment chain
  in all three languages.)
- Not deduplicating edges when the same fragment appears twice in the input (interviewers
  occasionally insert this). Indexing edges by input position handles it without a dedupe step.

**Preparation:**

- Implement Hierholzer iteratively on paper before the loop; the recursive version is easy to
  memorize and easy to fail under pressure.
- Drill the directed Part 1 in under 8 minutes using a single hashmap walk so you bank time
  for Parts 2-3.
- Pre-rehearse the question "does payload flip when the edge is traversed in reverse?" —
  asking it costs nothing and saves you from rewriting Part 2.
- Read the in-memory DB and beam search prompts in the same loop family; they share the
  multi-follow-up cadence.

---

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

**Java Solution:**

```java
import java.util.*;
import java.util.stream.Collectors;

public class InMemoryDB {

    public record Cell(String raw, boolean quoted) {}
    public record Predicate(String column, String op, Object value) {}
    public record Order(String column, boolean desc) {}
    public record Agg(String fn, String column) {}

    private final List<String> columns = new ArrayList<>();
    private final List<Map<String, Object>> rows = new ArrayList<>();

    public InMemoryDB(String csvText) {
        List<List<Cell>> parsed = parseCsv(csvText);
        if (parsed.isEmpty()) return;
        for (Cell c : parsed.get(0)) columns.add(c.raw());
        for (int r = 1; r < parsed.size(); r++) {
            List<Cell> cells = parsed.get(r);
            if (cells.size() != columns.size()) {
                throw new IllegalArgumentException("row has " + cells.size()
                        + " cells, header has " + columns.size());
            }
            Map<String, Object> row = new LinkedHashMap<>();
            for (int i = 0; i < columns.size(); i++) {
                row.put(columns.get(i), coerce(cells.get(i)));
            }
            rows.add(row);
        }
    }

    // ---------- CSV state machine: O(C) in characters ----------
    private static final int FIELD_START = 0, IN_FIELD = 1, IN_QUOTED = 2, AFTER_QUOTE = 3;

    public static List<List<Cell>> parseCsv(String text) {
        List<List<Cell>> out = new ArrayList<>();
        List<Cell> row = new ArrayList<>();
        StringBuilder buf = new StringBuilder();
        boolean quoted = false;
        int state = FIELD_START;

        for (int i = 0; i < text.length(); i++) {
            char ch = text.charAt(i);
            switch (state) {
                case FIELD_START -> {
                    if (ch == '"') { quoted = true; state = IN_QUOTED; }
                    else if (ch == ',') { row.add(makeCell(buf, quoted)); quoted = false; }
                    else if (ch == '\n') {
                        row.add(makeCell(buf, quoted)); quoted = false;
                        out.add(row); row = new ArrayList<>();
                    } else if (ch != '\r') { buf.append(ch); state = IN_FIELD; }
                }
                case IN_FIELD -> {
                    if (ch == ',') {
                        row.add(makeCell(buf, quoted)); quoted = false; state = FIELD_START;
                    } else if (ch == '\n') {
                        row.add(makeCell(buf, quoted)); quoted = false; state = FIELD_START;
                        out.add(row); row = new ArrayList<>();
                    } else if (ch != '\r') buf.append(ch);
                }
                case IN_QUOTED -> {
                    if (ch == '"') state = AFTER_QUOTE;
                    else buf.append(ch);              // commas and newlines are literal here
                }
                default -> {                           // AFTER_QUOTE
                    if (ch == '"') { buf.append('"'); state = IN_QUOTED; }   // doubled quote
                    else if (ch == ',') {
                        row.add(makeCell(buf, quoted)); quoted = false; state = FIELD_START;
                    } else if (ch == '\n') {
                        row.add(makeCell(buf, quoted)); quoted = false; state = FIELD_START;
                        out.add(row); row = new ArrayList<>();
                    }
                    // anything else after a closing quote is stray junk: ignore
                }
            }
        }
        if (buf.length() > 0 || !row.isEmpty() || state != FIELD_START) {
            row.add(makeCell(buf, quoted));
            out.add(row);
        }
        out.removeIf(r -> r.size() == 1 && r.get(0).raw().isEmpty() && !r.get(0).quoted());
        return out;
    }

    private static Cell makeCell(StringBuilder buf, boolean quoted) {
        String raw = buf.toString();
        buf.setLength(0);
        return new Cell(quoted ? raw : raw.strip(), quoted);   // trim only unquoted cells
    }

    /** Quoted cells stay String; unquoted numeric-looking cells become Long/Double. */
    public static Object coerce(Cell cell) {
        if (cell.quoted()) return cell.raw();
        try { return Long.parseLong(cell.raw()); } catch (NumberFormatException ignored) { }
        try { return Double.parseDouble(cell.raw()); } catch (NumberFormatException ignored) { }
        return cell.raw();
    }

    // ---------- comparison ----------
    private static Double numeric(Object v) {
        if (v instanceof Number n) return n.doubleValue();
        if (v instanceof String s) {
            try { return Double.parseDouble(s.strip()); }
            catch (NumberFormatException e) { return null; }
        }
        return null;
    }

    public static int compare(Object a, Object b) {
        Double na = numeric(a), nb = numeric(b);
        if (na != null && nb != null) return Double.compare(na, nb);
        if (a instanceof String sa && b instanceof String sb) return sa.compareTo(sb);
        throw new ClassCastException("cannot compare " + a + " with " + b);
    }

    private static boolean apply(String op, int c) {
        return switch (op) {
            case "==" -> c == 0;
            case "!=" -> c != 0;
            case "<"  -> c < 0;
            case "<=" -> c <= 0;
            case ">"  -> c > 0;
            case ">=" -> c >= 0;
            default -> throw new IllegalArgumentException("unknown operator " + op);
        };
    }

    // ---------- Parts 1-3: filter -> sort -> project ----------
    public List<Map<String, Object>> select(List<String> cols,
                                            List<Predicate> where,
                                            List<Order> orderBy) {
        List<String> projection = cols == null ? columns : cols;
        projection.forEach(this::require);

        List<Map<String, Object>> filtered = rows.stream()
                .filter(r -> matches(r, where))
                .collect(Collectors.toCollection(ArrayList::new));

        if (orderBy != null && !orderBy.isEmpty()) {
            orderBy.forEach(o -> require(o.column()));
            filtered.sort((a, b) -> {              // List.sort is stable -> ties keep input order
                for (Order o : orderBy) {
                    int c = compare(a.get(o.column()), b.get(o.column()));
                    if (c != 0) return o.desc() ? -c : c;
                }
                return 0;
            });
        }

        List<Map<String, Object>> out = new ArrayList<>();
        for (Map<String, Object> r : filtered) {
            Map<String, Object> rec = new LinkedHashMap<>();
            for (String c : projection) rec.put(c, r.get(c));
            out.add(rec);
        }
        return out;
    }

    public List<Map<String, Object>> select(List<String> cols) {
        return select(cols, null, null);
    }

    // ---------- Part 4: aggregation, single streaming pass ----------
    public List<Map<String, Object>> aggregate(List<Agg> aggregates,
                                               List<Predicate> where,
                                               List<String> groupBy) {
        List<String> keys = groupBy == null ? List.of() : groupBy;
        keys.forEach(this::require);
        Map<List<Object>, Object[]> state = new LinkedHashMap<>();

        for (Map<String, Object> row : rows) {
            if (!matches(row, where)) continue;
            List<Object> key = keys.stream().map(row::get).collect(Collectors.toList());
            Object[] acc = state.computeIfAbsent(key, k -> new Object[aggregates.size()]);
            for (int i = 0; i < aggregates.size(); i++) {
                acc[i] = step(aggregates.get(i), acc[i], row);
            }
        }

        List<Map<String, Object>> out = new ArrayList<>();
        for (var e : state.entrySet()) {
            Map<String, Object> rec = new LinkedHashMap<>();
            for (int i = 0; i < keys.size(); i++) rec.put(keys.get(i), e.getKey().get(i));
            for (int i = 0; i < aggregates.size(); i++) {
                Agg a = aggregates.get(i);
                rec.put(a.fn().toUpperCase() + "(" + a.column() + ")",
                        finalizeAgg(a, e.getValue()[i]));
            }
            out.add(rec);
        }
        return out;
    }

    private Object step(Agg agg, Object cur, Map<String, Object> row) {
        String fn = agg.fn().toUpperCase();
        if (fn.equals("COUNT")) return (cur == null ? 0L : (Long) cur) + 1;
        require(agg.column());
        Object v = row.get(agg.column());
        switch (fn) {
            case "SUM", "AVG" -> {
                Double num = numeric(v);
                if (num == null) throw new ClassCastException(fn + " over non-numeric " + v);
                double[] st = cur == null ? new double[]{0, 0} : (double[]) cur;
                return new double[]{st[0] + num, st[1] + 1};   // (total, count)
            }
            case "MIN" -> { return cur == null || compare(v, cur) < 0 ? v : cur; }
            case "MAX" -> { return cur == null || compare(v, cur) > 0 ? v : cur; }
            default -> throw new IllegalArgumentException("unknown aggregate " + fn);
        }
    }

    private Object finalizeAgg(Agg agg, Object cur) {
        String fn = agg.fn().toUpperCase();
        switch (fn) {
            case "COUNT" -> { return cur == null ? 0L : cur; }
            case "SUM" -> {
                double total = cur == null ? 0 : ((double[]) cur)[0];
                return total == Math.rint(total) ? (Object) (long) total : (Object) total;
            }
            case "AVG" -> {
                if (cur == null) return null;
                double[] st = (double[]) cur;
                return st[1] == 0 ? null : st[0] / st[1];
            }
            default -> { return cur; }
        }
    }

    // ---------- Part 5: insert / update / delete ----------
    public void insert(Map<String, Object> row) {
        for (String c : row.keySet()) require(c);
        Map<String, Object> rec = new LinkedHashMap<>();
        for (String c : columns) rec.put(c, row.getOrDefault(c, ""));
        rows.add(rec);
    }

    public int update(Map<String, Object> assignments, List<Predicate> where) {
        assignments.keySet().forEach(this::require);
        int n = 0;
        for (Map<String, Object> row : rows) {
            if (matches(row, where)) { row.putAll(assignments); n++; }
        }
        return n;
    }

    public int delete(List<Predicate> where) {
        int before = rows.size();
        rows.removeIf(r -> matches(r, where));
        return before - rows.size();
    }

    // ---------- helpers ----------
    public List<String> getColumns() { return columns; }

    private void require(String col) {
        if (!columns.contains(col)) throw new NoSuchElementException("unknown column " + col);
    }

    private boolean matches(Map<String, Object> row, List<Predicate> where) {
        if (where == null) return true;
        for (Predicate p : where) {          // conjunctive: every predicate must hold
            require(p.column());
            if (!apply(p.op(), compare(row.get(p.column()), p.value()))) return false;
        }
        return true;
    }

    public static void main(String[] args) {
        String csv =
                "Key,location,weather,temperature,data\n"
              + "1,\"Sunnyvale\",\"sunny\",100,\"datetimestamp\"\n"
              + "2,\"Seattle\",\"rain, heavy\",48,\"datetimestamp\"\n"
              + "3,\"Redmond\",\"he said \"\"cold\"\"\",30,\"datetimestamp\"\n"
              + "4,\"Sunnyvale\",\"sunny\",72,\"datetimestamp\"\n";
        InMemoryDB db = new InMemoryDB(csv);

        System.out.println(db.select(List.of("location", "temperature"),
                List.of(new Predicate("temperature", ">", 50)), null));
        // [{location=Sunnyvale, temperature=100}, {location=Sunnyvale, temperature=72}]

        System.out.println(db.select(List.of("Key"), null,
                List.of(new Order("location", false), new Order("temperature", true))));
        // [{Key=3}, {Key=2}, {Key=1}, {Key=4}]

        System.out.println(db.aggregate(
                List.of(new Agg("AVG", "temperature"), new Agg("COUNT", "*")),
                null, List.of("location")));
        // [{location=Sunnyvale, AVG(temperature)=86.0, COUNT(*)=2}, ...]
    }
}
```

**TypeScript Solution:**

```typescript
type Value = number | string;
type Row = Record<string, Value>;
type Predicate = [string, string, Value];
type Order = [string, "ASC" | "DESC"];
type Agg = [string, string];
interface Cell { raw: string; quoted: boolean; }

// ---------- CSV state machine: O(C) in characters ----------
const S = { FieldStart: 0, InField: 1, InQuoted: 2, AfterQuote: 3 } as const;

function parseCsv(text: string): Cell[][] {
    const out: Cell[][] = [];
    let row: Cell[] = [];
    let buf = "";
    let quoted = false;
    let state: number = S.FieldStart;

    const endField = () => {
        row.push({ raw: quoted ? buf : buf.trim(), quoted });   // trim only unquoted cells
        buf = "";
        quoted = false;
        state = S.FieldStart;
    };
    const endRow = () => { endField(); out.push(row); row = []; };

    for (const ch of text) {
        switch (state) {
            case S.FieldStart:
                if (ch === '"') { quoted = true; state = S.InQuoted; }
                else if (ch === ",") endField();
                else if (ch === "\n") endRow();
                else if (ch !== "\r") { buf += ch; state = S.InField; }
                break;
            case S.InField:
                if (ch === ",") endField();
                else if (ch === "\n") endRow();
                else if (ch !== "\r") buf += ch;
                break;
            case S.InQuoted:
                if (ch === '"') state = S.AfterQuote;
                else buf += ch;                      // commas and newlines are literal here
                break;
            case S.AfterQuote:
                if (ch === '"') { buf += '"'; state = S.InQuoted; }   // doubled quote
                else if (ch === ",") endField();
                else if (ch === "\n") endRow();
                break;                                // stray junk after a close quote: ignore
        }
    }
    if (buf || row.length || state !== S.FieldStart) endRow();
    return out.filter((r) => !(r.length === 1 && r[0].raw === "" && !r[0].quoted));
}

/** Quoted cells stay string; unquoted numeric-looking cells become number. */
function coerce({ raw, quoted }: Cell): Value {
    if (quoted) return raw;
    if (raw !== "" && Number.isFinite(Number(raw))) return Number(raw);
    return raw;
}

function numeric(v: Value): number | null {
    if (typeof v === "number") return v;
    if (v.trim() !== "" && Number.isFinite(Number(v))) return Number(v);
    return null;
}

function compare(a: Value, b: Value): number {
    const na = numeric(a), nb = numeric(b);
    if (na !== null && nb !== null) return na === nb ? 0 : na < nb ? -1 : 1;
    if (typeof a === "string" && typeof b === "string") return a === b ? 0 : a < b ? -1 : 1;
    throw new TypeError(`cannot compare ${JSON.stringify(a)} with ${JSON.stringify(b)}`);
}

const OPS: Record<string, (c: number) => boolean> = {
    "==": (c) => c === 0,
    "!=": (c) => c !== 0,
    "<": (c) => c < 0,
    "<=": (c) => c <= 0,
    ">": (c) => c > 0,
    ">=": (c) => c >= 0,
};

class InMemoryDB {
    columns: string[] = [];
    rows: Row[] = [];

    constructor(csvText: string) {
        const parsed = parseCsv(csvText);
        if (!parsed.length) return;
        this.columns = parsed[0].map((c) => c.raw);
        for (const cells of parsed.slice(1)) {
            if (cells.length !== this.columns.length) {
                throw new Error(`row has ${cells.length} cells, `
                    + `header has ${this.columns.length}`);
            }
            const row: Row = {};
            this.columns.forEach((c, i) => { row[c] = coerce(cells[i]); });
            this.rows.push(row);
        }
    }

    // ---------- Parts 1-3: filter -> sort -> project ----------
    select(columns?: string[] | null, where?: Predicate[], orderBy?: Order[]): Row[] {
        const cols = columns ?? this.columns;
        cols.forEach((c) => this.require(c));

        let rows = this.rows.filter((r) => this.matches(r, where));
        if (orderBy?.length) {
            orderBy.forEach(([c]) => this.require(c));
            rows = rows.slice().sort((a, b) => {   // Array#sort is stable in ES2019+
                for (const [col, dir] of orderBy) {
                    const c = compare(a[col], b[col]);
                    if (c) return dir.toUpperCase() === "DESC" ? -c : c;
                }
                return 0;
            });
        }
        return rows.map((r) => Object.fromEntries(cols.map((c) => [c, r[c]])) as Row);
    }

    // ---------- Part 4: aggregation, single streaming pass ----------
    aggregate(aggregates: Agg[], where?: Predicate[], groupBy?: string[]): Row[] {
        const keys = groupBy ?? [];
        keys.forEach((c) => this.require(c));
        const state = new Map<string, { key: Value[]; acc: unknown[] }>();

        for (const row of this.rows) {
            if (!this.matches(row, where)) continue;
            const key = keys.map((c) => row[c]);
            const id = JSON.stringify(key);
            if (!state.has(id)) {
                state.set(id, { key, acc: new Array(aggregates.length).fill(null) });
            }
            const entry = state.get(id)!;
            aggregates.forEach(([fn, col], i) => {
                entry.acc[i] = this.step(fn, col, entry.acc[i], row);
            });
        }

        return [...state.values()].map(({ key, acc }) => {
            const rec: Row = Object.fromEntries(keys.map((c, i) => [c, key[i]]));
            aggregates.forEach(([fn, col], i) => {
                rec[`${fn.toUpperCase()}(${col})`] =
                    InMemoryDB.finalize(fn.toUpperCase(), acc[i]);
            });
            return rec;
        });
    }

    private step(fnRaw: string, col: string, cur: unknown, row: Row): unknown {
        const fn = fnRaw.toUpperCase();
        if (fn === "COUNT") return ((cur as number) ?? 0) + 1;
        this.require(col);
        const v = row[col];
        if (fn === "SUM" || fn === "AVG") {
            const num = numeric(v);
            if (num === null) throw new TypeError(`${fn} over non-numeric value ${v}`);
            const [total, count] = (cur as [number, number]) ?? [0, 0];   // (total, count)
            return [total + num, count + 1];
        }
        if (fn === "MIN") return cur === null || compare(v, cur as Value) < 0 ? v : cur;
        if (fn === "MAX") return cur === null || compare(v, cur as Value) > 0 ? v : cur;
        throw new Error(`unknown aggregate ${fn}`);
    }

    private static finalize(fn: string, cur: unknown): Value {
        if (fn === "COUNT") return (cur as number) ?? 0;
        if (fn === "SUM") return ((cur as [number, number]) ?? [0, 0])[0];
        if (fn === "AVG") {
            const [total, count] = (cur as [number, number]) ?? [0, 0];
            return count ? total / count : 0;
        }
        return cur as Value;
    }

    // ---------- Part 5: insert / update / delete ----------
    insert(row: Row): void {
        Object.keys(row).forEach((c) => this.require(c));
        this.rows.push(Object.fromEntries(this.columns.map((c) => [c, row[c] ?? ""])) as Row);
    }

    update(assignments: Row, where?: Predicate[]): number {
        Object.keys(assignments).forEach((c) => this.require(c));
        let n = 0;
        for (const row of this.rows) {
            if (this.matches(row, where)) { Object.assign(row, assignments); n++; }
        }
        return n;
    }

    delete(where?: Predicate[]): number {
        const keep = this.rows.filter((r) => !this.matches(r, where));
        const removed = this.rows.length - keep.length;
        this.rows = keep;
        return removed;
    }

    // ---------- helpers ----------
    private require(col: string): void {
        if (!this.columns.includes(col)) throw new Error(`unknown column ${col}`);
    }

    private matches(row: Row, where?: Predicate[]): boolean {
        for (const [col, op, target] of where ?? []) {   // conjunctive
            this.require(col);
            const fn = OPS[op];
            if (!fn) throw new Error(`unknown operator ${op}`);
            if (!fn(compare(row[col], target))) return false;
        }
        return true;
    }
}

// Example usage
const CSV =
    'Key,location,weather,temperature,data\n' +
    '1,"Sunnyvale","sunny",100,"datetimestamp"\n' +
    '2,"Seattle","rain, heavy",48,"datetimestamp"\n' +
    '3,"Redmond","he said ""cold""",30,"datetimestamp"\n' +
    '4,"Sunnyvale","sunny",72,"datetimestamp"\n';

const db = new InMemoryDB(CSV);

console.log(db.select(["location", "temperature"], [["temperature", ">", 50]]));
// [ { location: 'Sunnyvale', temperature: 100 },
//   { location: 'Sunnyvale', temperature: 72 } ]

console.log(db.select(["Key"], undefined, [["location", "ASC"], ["temperature", "DESC"]]));
// [ { Key: 3 }, { Key: 2 }, { Key: 1 }, { Key: 4 } ]

console.log(db.aggregate([["AVG", "temperature"], ["COUNT", "*"]], undefined, ["location"]));
// [ { location: 'Sunnyvale', 'AVG(temperature)': 86, 'COUNT(*)': 2 }, ... ]
```

**Preparation:**

- Pre-write and memorize a 30-line CSV state-machine parser. Practice it on `a,"b,c",d` and
  `"he said ""hi"""` until you can type it without thinking.
- Implement the full SELECT / WHERE / ORDER BY pipeline once end-to-end on paper before the
  loop; time yourself at 25 minutes for the full skeleton.
- For the aggregation variant, hold a single dict of `column → running_state` per group and
  finalize once at the end — don't materialize groups.
- Budget time: 15 min parser, 10 min Part 1, 10 min Part 2, 10 min Part 3, 15 min for
  whatever comes next.
- The same prompt family appears across other major AI labs — pattern-match it as the
  "OpenAI in-memory DB problem with a CSV-input twist". Recruiters for MAI Copilot explicitly
  recommend rehearsing that bank of problems before the loop.

---

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

**Java Solution:**

```java
import java.util.*;
import java.util.concurrent.locks.ReentrantLock;

public class LRUCache {

    static class Node {
        int key, value;
        Node prev, next;
        Node(int key, int value) { this.key = key; this.value = value; }
    }

    private final int capacity;
    private final Map<Integer, Node> map = new HashMap<>();
    private final Node head = new Node(0, 0);   // MRU sentinel
    private final Node tail = new Node(0, 0);   // LRU sentinel

    public LRUCache(int capacity) {
        if (capacity <= 0) throw new IllegalArgumentException("capacity must be positive");
        this.capacity = capacity;
        head.next = tail;                        // sentinels remove every splicing edge case
        tail.prev = head;
    }

    private void unlink(Node node) {
        node.prev.next = node.next;
        node.next.prev = node.prev;
    }

    private void pushFront(Node node) {
        node.prev = head;
        node.next = head.next;
        head.next.prev = node;
        head.next = node;
    }

    public int get(int key) {
        Node node = map.get(key);
        if (node == null) return -1;
        unlink(node);                            // a read counts as a use
        pushFront(node);
        return node.value;
    }

    public void put(int key, int value) {
        Node node = map.get(key);
        if (node != null) {                      // update: refresh recency, never evict
            node.value = value;
            unlink(node);
            pushFront(node);
            return;
        }
        if (map.size() == capacity) {
            Node lru = tail.prev;
            unlink(lru);
            map.remove(lru.key);
        }
        node = new Node(key, value);
        map.put(key, node);
        pushFront(node);
    }

    public int size() { return map.size(); }

    public List<Integer> keysMruToLru() {
        List<Integer> out = new ArrayList<>();
        for (Node cur = head.next; cur != tail; cur = cur.next) out.add(cur.key);
        return out;
    }

    /**
     * Follow-up 2: one lock around every public method.
     * get() mutates the recency list, so it is a writer — an RWLock buys nothing.
     */
    public static class ThreadSafe extends LRUCache {
        private final ReentrantLock lock = new ReentrantLock();

        public ThreadSafe(int capacity) { super(capacity); }

        @Override public int get(int key) {
            lock.lock();
            try { return super.get(key); } finally { lock.unlock(); }
        }

        @Override public void put(int key, int value) {
            lock.lock();
            try { super.put(key, value); } finally { lock.unlock(); }
        }

        @Override public int size() {
            lock.lock();
            try { return super.size(); } finally { lock.unlock(); }
        }

        @Override public List<Integer> keysMruToLru() {
            lock.lock();
            try { return super.keysMruToLru(); } finally { lock.unlock(); }
        }
    }

    /** Follow-up 2b: shard to cut contention (the ConcurrentHashMap pattern). */
    public static class Sharded {
        private final ThreadSafe[] shards;

        public Sharded(int capacity, int shardCount) {
            shardCount = Math.max(1, Math.min(shardCount, capacity));
            shards = new ThreadSafe[shardCount];
            int base = capacity / shardCount, extra = capacity % shardCount;
            for (int i = 0; i < shardCount; i++) {
                shards[i] = new ThreadSafe(base + (i < extra ? 1 : 0));
            }
        }

        private ThreadSafe shard(int key) {
            return shards[Math.floorMod(Integer.hashCode(key), shards.length)];
        }

        public int get(int key) { return shard(key).get(key); }
        public void put(int key, int value) { shard(key).put(key, value); }
    }

    /** Follow-up 3: LFU — evict least frequently used, LRU within a frequency tie. */
    public static class LFUCache {
        private final int capacity;
        private final Map<Integer, Integer> values = new HashMap<>();
        private final Map<Integer, Integer> freq = new HashMap<>();
        // LinkedHashSet keeps insertion order, so each bucket is its own LRU queue.
        private final Map<Integer, LinkedHashSet<Integer>> buckets = new HashMap<>();
        private int minFreq = 0;

        public LFUCache(int capacity) { this.capacity = capacity; }

        private void bump(int key) {
            int f = freq.get(key);
            LinkedHashSet<Integer> bucket = buckets.get(f);
            bucket.remove(key);
            if (bucket.isEmpty()) {
                buckets.remove(f);
                if (minFreq == f) minFreq = f + 1;   // only way minFreq ever advances
            }
            freq.put(key, f + 1);
            buckets.computeIfAbsent(f + 1, k -> new LinkedHashSet<>()).add(key);
        }

        public int get(int key) {
            if (!values.containsKey(key)) return -1;
            bump(key);
            return values.get(key);
        }

        public void put(int key, int value) {
            if (capacity <= 0) return;
            if (values.containsKey(key)) {
                values.put(key, value);
                bump(key);
                return;
            }
            if (values.size() == capacity) {
                LinkedHashSet<Integer> bucket = buckets.get(minFreq);
                int victim = bucket.iterator().next();    // insertion order = LRU first
                bucket.remove(victim);
                if (bucket.isEmpty()) buckets.remove(minFreq);
                values.remove(victim);
                freq.remove(victim);
            }
            values.put(key, value);
            freq.put(key, 1);
            buckets.computeIfAbsent(1, k -> new LinkedHashSet<>()).add(key);
            minFreq = 1;                                  // a fresh insert resets minFreq
        }
    }

    public static void main(String[] args) {
        LRUCache cache = new LRUCache(2);
        cache.put(1, 1);
        cache.put(2, 2);
        System.out.println(cache.get(1));   // 1
        cache.put(3, 3);                    // evicts key 2
        System.out.println(cache.get(2));   // -1

        LFUCache lfu = new LFUCache(2);
        lfu.put(1, 1);
        lfu.put(2, 2);
        lfu.get(1);                         // freq: 1 -> 2, 2 -> 1
        lfu.put(3, 3);                      // evicts key 2
        System.out.println(lfu.get(2));     // -1
    }
}
```

**TypeScript Solution:**

```typescript
class ListNode {
    key: number;
    value: number;
    prev: ListNode | null = null;
    next: ListNode | null = null;
    constructor(key = 0, value = 0) { this.key = key; this.value = value; }
}

/**
 * hashmap + doubly-linked list.
 * head sentinel = MRU side, tail sentinel = LRU side.
 * get/put: O(1). Space: O(capacity).
 */
class LRUCache {
    private readonly capacity: number;
    private readonly map = new Map<number, ListNode>();
    private readonly head = new ListNode();   // sentinels remove every splicing edge case
    private readonly tail = new ListNode();

    constructor(capacity: number) {
        if (capacity <= 0) throw new Error("capacity must be positive");
        this.capacity = capacity;
        this.head.next = this.tail;
        this.tail.prev = this.head;
    }

    private unlink(node: ListNode): void {
        node.prev!.next = node.next;
        node.next!.prev = node.prev;
    }

    private pushFront(node: ListNode): void {
        node.prev = this.head;
        node.next = this.head.next;
        this.head.next!.prev = node;
        this.head.next = node;
    }

    get(key: number): number {
        const node = this.map.get(key);
        if (!node) return -1;
        this.unlink(node);                    // a read counts as a use
        this.pushFront(node);
        return node.value;
    }

    put(key: number, value: number): void {
        const existing = this.map.get(key);
        if (existing) {                       // update: refresh recency, never evict
            existing.value = value;
            this.unlink(existing);
            this.pushFront(existing);
            return;
        }
        if (this.map.size === this.capacity) {
            const lru = this.tail.prev!;
            this.unlink(lru);
            this.map.delete(lru.key);
        }
        const node = new ListNode(key, value);
        this.map.set(key, node);
        this.pushFront(node);
    }

    get size(): number { return this.map.size; }

    keysMruToLru(): number[] {
        const out: number[] = [];
        for (let cur = this.head.next!; cur !== this.tail; cur = cur.next!) out.push(cur.key);
        return out;
    }
}

/**
 * JS is single-threaded, so there is no data race inside one event-loop turn;
 * the equivalent of the RLock follow-up is serializing async work behind a
 * promise chain so an await inside a critical section cannot interleave.
 */
class AsyncLockedLRUCache {
    private readonly inner: LRUCache;
    private queue: Promise<unknown> = Promise.resolve();

    constructor(capacity: number) { this.inner = new LRUCache(capacity); }

    private run<T>(fn: () => T): Promise<T> {
        const result = this.queue.then(fn, fn);
        this.queue = result.catch(() => undefined);   // one rejection must not stall the queue
        return result;
    }

    get(key: number): Promise<number> { return this.run(() => this.inner.get(key)); }
    put(key: number, value: number): Promise<void> {
        return this.run(() => this.inner.put(key, value));
    }
}

/** Shard to spread work (and, in a worker-thread port, lock contention). */
class ShardedLRUCache {
    private readonly shards: LRUCache[];

    constructor(capacity: number, shardCount = 16) {
        const n = Math.max(1, Math.min(shardCount, capacity));
        const base = Math.floor(capacity / n), extra = capacity % n;
        this.shards = Array.from({ length: n },
            (_, i) => new LRUCache(base + (i < extra ? 1 : 0)));
    }

    private shard(key: number): LRUCache {
        let h = key | 0;
        h = (h ^ (h >>> 16)) >>> 0;           // spread low bits, like HashMap#hash
        return this.shards[h % this.shards.length];
    }

    get(key: number): number { return this.shard(key).get(key); }
    put(key: number, value: number): void { this.shard(key).put(key, value); }
}

/** LFU: freq -> insertion-ordered bucket; ties inside a bucket break LRU. */
class LFUCache {
    private readonly capacity: number;
    private readonly values = new Map<number, number>();
    private readonly freq = new Map<number, number>();
    private readonly buckets = new Map<number, Set<number>>();   // Set keeps insertion order
    private minFreq = 0;

    constructor(capacity: number) { this.capacity = capacity; }

    private bump(key: number): void {
        const f = this.freq.get(key)!;
        const bucket = this.buckets.get(f)!;
        bucket.delete(key);
        if (bucket.size === 0) {
            this.buckets.delete(f);
            if (this.minFreq === f) this.minFreq = f + 1;   // only way minFreq advances
        }
        this.freq.set(key, f + 1);
        if (!this.buckets.has(f + 1)) this.buckets.set(f + 1, new Set());
        this.buckets.get(f + 1)!.add(key);
    }

    get(key: number): number {
        if (!this.values.has(key)) return -1;
        this.bump(key);
        return this.values.get(key)!;
    }

    put(key: number, value: number): void {
        if (this.capacity <= 0) return;
        if (this.values.has(key)) {
            this.values.set(key, value);
            this.bump(key);
            return;
        }
        if (this.values.size === this.capacity) {
            const bucket = this.buckets.get(this.minFreq)!;
            const victim = bucket.values().next().value as number;   // oldest in lowest bucket
            bucket.delete(victim);
            if (bucket.size === 0) this.buckets.delete(this.minFreq);
            this.values.delete(victim);
            this.freq.delete(victim);
        }
        this.values.set(key, value);
        this.freq.set(key, 1);
        if (!this.buckets.has(1)) this.buckets.set(1, new Set());
        this.buckets.get(1)!.add(key);
        this.minFreq = 1;                     // a fresh insert resets minFreq
    }
}

// Example usage
const cache = new LRUCache(2);
cache.put(1, 1);
cache.put(2, 2);
console.log(cache.get(1));   // 1
cache.put(3, 3);             // evicts key 2
console.log(cache.get(2));   // -1

const lfu = new LFUCache(2);
lfu.put(1, 1);
lfu.put(2, 2);
lfu.get(1);                  // freq: 1 -> 2, 2 -> 1
lfu.put(3, 3);               // evicts key 2
console.log(lfu.get(2));     // -1
```

**Preparation:**

- Write the LRU skeleton from scratch under 12 minutes; type the sentinel-node pattern until
  it is automatic.
- Drill the four splice operations (unlink, push-to-head, unlink-tail, pop-from-dict) on paper.
- Pre-rehearse the thread-safety extension: "wrap with RLock for correctness; shard for
  throughput" — and be ready to explain why an RWLock does *not* apply.
- For LFU, write it once end-to-end the night before; the bucket bookkeeping is the part
  candidates lose time on.

---

## References

- LeetCode 146 — LRU Cache; LeetCode 460 — LFU Cache
- Hierholzer's algorithm for Eulerian paths (Part 2 of the shotgun sequencing problem)
- RFC 4180 — the CSV conventions the in-memory DB parser has to honor
