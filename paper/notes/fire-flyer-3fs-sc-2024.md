# Fire-Flyer AI-HPC and 3FS: A Cost-Effective Software-Hardware Co-Design for Deep Learning

> SC24 (also arXiv) plus the 3FS design notes — structured reading notes, storage-focused

Full paper content: [Markdown conversion](../original/fire-flyer-sc-2024.md)

## Paper information

- **Authors:** DeepSeek-AI team (long author list, alphabetical by surname)
- **Venue:** SC24, November 17-22, 2024, Atlanta, USA. arXiv: [2408.14158](https://arxiv.org/abs/2408.14158)
- **Code:** [deepseek-ai/3FS](https://github.com/deepseek-ai/3FS)
- **Second source used here:** the repository's
  [design notes](https://github.com/deepseek-ai/3FS/blob/main/docs/design_notes.md). I read the
  `main` branch version; it may change, and I did not vendor a copy. Everything under
  "3FS design in depth" below comes from it, **not** from the paper.
- **What the paper covers:** the whole Fire-Flyer 2 cluster: PCIe A100 nodes, network, the HFReduce
  all-reduce library, HaiScale training framework, the 3FS file system, the HAI scheduling
  platform, checkpointing, and hardware-failure data. **3FS gets about one page**. This note focuses
  on storage (3FS, the network it depends on, and checkpointing), and covers the rest briefly.

## One-sentence summary

DeepSeek built a 10,000-GPU training cluster from cheaper PCIe A100 nodes on a single two-layer
InfiniBand fat-tree shared by compute and storage, then co-designed software around it: HFReduce for
all-reduce, traffic isolation and request-to-send flow control to keep the shared network
congestion-free, and 3FS, a disaggregated all-flash file system (CRAQ chain replication over
SSDs, metadata in a transactional key-value store, an io_uring-style zero-copy client) that reads at
8 TB/s and makes saving a checkpoint every five minutes cheap.

## Vocabulary you need first

| Term | Meaning |
|---|---|
| **AI-HPC** | A cluster built for deep-learning training with HPC-style networking. |
| **PCIe A100 vs DGX/SXM A100** | Same GPU chip, different packaging: PCIe cards in a commodity server versus NVIDIA's NVLink-connected DGX board. PCIe is cheaper but has less inter-GPU bandwidth. |
| **Fat-tree** | A network topology with full bisection bandwidth: every host can talk to every other at full rate. |
| **IB / RoCE** | InfiniBand / RDMA over Converged Ethernet. Both let a NIC read or write remote memory without involving the remote CPU. |
| **RDMA read/write/send** | One-sided memory operations (read, write) and a message (send) over the network. |
| **Incast** | Many senders replying to one receiver at once, overflowing the receiver's switch port. |
| **Service Level (SL) / Virtual Lane (VL)** | InfiniBand mechanism to put traffic classes in separate queues so they don't block each other. |
| **CRAQ** | Chain Replication with Apportioned Queries: writes go down a chain of replicas, reads can be served by any replica. |
| **Chain, chain table, storage target** | 3FS terms, defined below. |
| **FUSE** | Filesystem in Userspace: lets a user-space daemon implement a file system behind the kernel's VFS. |
| **FoundationDB** | A distributed transactional key-value store with serializable snapshot isolation. 3FS keeps all file metadata there. |
| **MFU** | Model FLOPs utilization: fraction of peak compute actually used. |

## The problem

- Compute demand for deep learning grows about **10x per year**, while hardware FLOPs grow only
  3.0x every two years, DRAM bandwidth 1.6x and interconnect bandwidth 1.4x (the paper's Figure 2).
  So you need many more machines, and cost and energy become the binding constraint.
- One training job needs hundreds of GPUs and heavy storage and network use. Two system challenges:
  **efficiency** (MFU, which depends on parallelism strategy, communication, data loading) and
  **stability** (weeks-long runs where stragglers and hardware failures are routine).
- Traditional supercomputers (TianHe-2A, Fugaku and others) lack FP16/tensor support or suit
  FP64 workloads; big-company clusters use expensive DGX-class nodes; cloud rental for a two-year
  project can cost as much as buying a cluster.
- DeepSeek's answer: spend less on hardware (PCIe GPUs, one NIC per node, fewer switches) and more
  on software.

## Hardware and network architecture

### Compute node (Fire-Flyer 2, built 2021)

- **8x A100 PCIe GPUs and 1x Mellanox ConnectX-6 200 Gbps IB NIC**, all attached directly to the CPU
  without a PCIe switch. The NIC sits on its own PCIe root complex so it doesn't contend with GPUs.
  Their workloads' storage plus communication bandwidth fits in one 200 Gbps NIC.
- An **NVLink Bridge** between GPU pairs (600 GB/s) was added later for LLM tensor parallelism.

### One network for compute *and* storage

- **Two-layer fat-tree**, Mellanox QM8700 switches (40 ports of 200 Gbps). One such tree supports up
  to 800 nodes (20 spine and 40 leaf switches).
- Cluster: about **1,250 GPU nodes** (10,000 GPUs) and nearly **200 storage servers**. To avoid a
  three-layer tree, the network is **two zones**, each an 800-port fat-tree with about 600 GPU nodes.
  Each storage server has **two IB NICs, one to each zone**, so every compute node reaches the
  shared storage. The zones are linked by a limited number of links, and the HAI scheduler permits
  **at most one cross-zone task**. Collectives (HFReduce or NCCL) cross zones through a double binary
  tree where only one pair of nodes talks across the zones.
- Dragonfly was rejected for lacking enough bisection bandwidth for integrated storage and compute.
  InfiniBand was chosen because RoCE was less mature at the time.

### Cost and performance (Tables II and III)

| Metric | PCIe A100 (Fire-Flyer 2) | DGX-A100 |
|---|---|---|
| TF32 GEMM per GPU | 107 TFLOPS | 131 |
| FP16 GEMM per GPU | 220 TFLOPS | 263 |
| Relative performance | 83% | 100% |
| Node relative price | 60% | 100% |
| Power per node | 2,500 W | 4,200 W |

| | Fire-Flyer two-zone | Same PCIe nodes, 3-layer fat-tree | DGX with 3-layer fat-tree |
|---|---|---|---|
| Switches | 122 | 200 | 1,320 |
| Network price (relative units) | 350 | 600 | 4,000 |
| Server price | 11,250 | 11,250 | 19,000 |
| **Total price** | **11,600** | 11,850 | 23,000 |

Headline: about 80% of DGX-A100 performance at about 60% of the cost (network saving of 40% versus
a comparable three-layer tree), and 40% less energy. The paper's total cluster energy is "just over
3 MW" (under 4 MW).

## HFReduce, HaiScale and the rest (brief)

Included because storage design depends on the same shared network.

- **HFReduce** replaces NCCL all-reduce on PCIe machines. Steps: (1) copy each GPU's gradient chunk
  asynchronously to CPU memory (GDRCopy for small, `MemcpyAsync` for large); (2) CPU adds the 8
  gradients with SIMD (FP32/FP16/BF16/FP8); (3) inter-node all-reduce of the reduced data on the CPU
  using a **double binary tree** over RDMA writes; (4) copy the result back to the GPUs.
- **Why it beats NCCL here:** NCCL's ring moves each byte over PCIe `2(n-1)/n` times, while HFReduce
  needs just one D2H and one H2D. It also uses the GPU copy engine, not GPU kernels, so it doesn't
  steal SMs from compute. The authors also found that EPYC Rome CPUs lack the PCIe "chained write"
  feature, capping GPU-to-NIC peer-to-peer at about 9 GiB/s and NCCL at about 4 GB/s.
- **Numbers:** inter-node bandwidth of **6.3-8.1 GB/s** for HFReduce versus **1.6-4.8 GB/s** for NCCL
  (186 MiB all-reduce), and over **10 GB/s** with the NVLink variant. Its own memory traffic is about
  24x the data size, bounding it at about 13.3 GB/s theoretical; they reach just over 8 GB/s.
- **HaiScale** (training library): DDP using HFReduce overlapped with backward; FSDP with overlap;
  pipeline/tensor/expert parallelism tuned for one NIC per 8 GPUs. Reported: VGG16 trains in about
  half the time of PyTorch DDP/NCCL, about 88% scalability 32 to 512 GPUs; LLaMA-13B 91% parallel
  efficiency 64 to 512 GPUs; DeepSeekMoE-16B 92.9% at 320 GPUs (76% at 640); GPT-2-medium FSDP 95%
  scalability 16 to 128 GPUs.
- **HAI Platform:** time-sharing scheduler. Jobs must handle an interrupt signal, save a checkpoint,
  notify the cluster, and resume from the checkpoint. Nodes (not pooled GPUs) are the unit; the
  authors cite 99% utilization.

## Keeping the shared network congestion-free (Section VI-A)

The trade-off of one network for everything is interference. Four kinds of traffic share it:
HFReduce, NCCL, 3FS storage, and other. Measures:

1. **Traffic isolation with InfiniBand Service Levels mapped to Virtual Lanes.** Each class gets its
   own queue and bandwidth proportion, preventing head-of-line blocking between classes.
2. **Static routing, not adaptive routing.** Storage creates many incast patterns; enabling adaptive
   routing "led to more severe congestion spread". Storage, compute and management nodes are spread
   evenly across leaf-to-spine links to disperse storage traffic.
3. **NCCL topology tuning:** route via the NIC and GPUs in the same NUMA node, and enable PCIe
   Relaxed Ordering.
4. **3FS request-to-send flow control** (below).

Discussion (Section VIII-A): they **disabled DCQCN** (the IB NIC congestion-control algorithm)
because no parameter set worked for HFReduce and 3FS traffic together. The network operates with no
congestion control, relying on the measures above. DPUs (BlueField) allow custom congestion control
(HPCC, TIMELY) but raise cost and operational complexity.

## 3FS in the paper (Section VI-B)

**Positioning.** 3FS is an in-house distributed file system "akin to WekaFS, DAOS and BeeGFS" whose
design focuses on fully using the IOPS and throughput of NVMe SSDs and RDMA.

**Storage hardware (Table IV).** Each node: 1x 64-core AMD EPYC 7742, 512 GB DDR4-3200 (8 channels),
**2x ConnectX-6 200 Gbps NICs**, **16x 15.36 TB PCIe 4.0 NVMe SSDs**. Deployment: **180 storage nodes**.

| Quantity | Value |
|---|---|
| Storage nodes | 180 |
| SSDs | 2,880 (180 x 16) |
| Outbound NICs | 360 x 200 Gbps, so about 9 TB/s theoretical |
| **Achieved aggregate read** | **8 TB/s** |
| Usable capacity | over 20 PiB, with mirrored (replicated) data |

(Raw capacity by my arithmetic is 2,880 x 15.36 TB, about 44 PB, so "over 20 PiB" is consistent
with keeping each chunk on multiple replicas.)

**Roles and layout.** Four roles: cluster manager, metadata service, storage service, client.
Metadata is held in a distributed key-value store with an inode table and a directory-entry table
keyed by (parent inode, name). Storage service implements **CRAQ** for strong consistency; files are
split into chunks replicated over a chain of **storage targets**; a **chain table** is an ordered set
of chains; the metadata service gives each file an offset in the chain table and a stripe size `k`,
and its chunks go to the next `k` chains. Each SSD serves several targets belonging to different
chains, which spreads load.

**Request-to-send (incast control).** Each 3FS client can reach every storage service, and under
peak load incast congestion appears on the client side. Mechanism: after a storage service reads
the requested data from SSD, it **asks the client for permission** to send; the client limits the
number of concurrent senders; once granted, the service sends with an RDMA WRITE followed by an RDMA
SEND to notify. This raises per-request latency but "is required to achieve sustainable high
throughput".

**3FS-KV.** A shared-storage distributed data processing layer on top of 3FS supporting three
models: key-value, message queue, and object storage, with read-write separation and on-demand
startup. It supports DeepSeek's "KV Context Caching on Disk", which the paper says reduces LLM
serving cost by an order of magnitude (no further detail or measurements in this paper).

## 3FS design in depth (from the repository's design notes)

*Everything in this section is from the design notes, not the SC24 paper.*

### Why a file system and not an object store

The authors argue object stores are popular for analytics and ML, but file semantics give more
flexibility:

- **Atomic directory manipulation.** Their common pattern: write files into a temp directory, then
  atomically move the directory to its final place. Also, with huge numbers of small files,
  recursive directory delete matters. Object stores approximate directories with slashes in keys and
  lack atomic move and recursive delete.
- **Symbolic and hard links** to make lightweight snapshots of datasets that grow by appending files.
- **Familiar interface:** existing CSV/Parquet data loaders just work with the FUSE or native client.

### The clients

- **FUSE client** (most applications): low adoption barrier but limited performance.
  - *Memory copies:* the user-space daemon can't access application memory, so data crosses
    kernel/user boundary.
  - *Lock contention:* FUSE puts requests in a multi-threaded shared queue under a spin lock;
    their benchmark shows FUSE tops out at about **400K 4 KiB reads/s** regardless of threads.
  - *No concurrent writes to the same file* on Linux 5.x; applications write multiple files
    concurrently instead.
  - Training jobs do random reads of a few KB to several MB per sample, not 4K aligned, so
    data loaders perform poorly on FUSE and don't fill SSD or RDMA bandwidth.
- **Why not a kernel module:** hard to develop and debug (a crash can leave no log), and upgrading
  requires stopping all processes using the file system or rebooting machines.
- **Native client with an asynchronous zero-copy API**, inspired by Linux `io_uring`, living inside
  the FUSE daemon:
  - Metadata operations (open, close, stat) still go through FUSE/POSIX; the application opens a file,
    gets an fd, and registers it with the native API. This keeps metadata semantics identical to POSIX.
  - **Iov:** a large memory region shared between the app and the native client for zero-copy reads
    and writes (the client manages InfiniBand memory registration). Reads land in the Iov; writes must
    be placed there before the call.
  - **Ior:** a small shared ring buffer used like an `io_uring` submission queue. The app enqueues
    read/write requests; the client dequeues them. Requests execute in batches sized by `io_depth`;
    batches from the same or different rings run in parallel. Multiple rings are recommended for
    multi-threaded apps because sharing a ring needs synchronization.
  - Native-client threads fetch requests from the Iors, **batch them and send them to storage
    services**, cutting RPC overhead for small reads.

### Metadata service and layout

- **File layout.** Per directory, users can set chain table, chunk size and stripe size. A chunk's id
  is the **file's inode id concatenated with the chunk index**. On file creation the meta service
  picks consecutive chains from the chain table by round-robin (per stripe size) and then **shuffles**
  them with a random seed, giving a balanced spread over chains and SSDs. On open, the client gets the
  layout once, then **computes chunk ids and chains itself**, keeping the meta service off the data path.
- **Metadata in FoundationDB.** Meta services are **stateless**, and any client can use any meta
  service; failed or timed-out requests fail over to another. Two structures:
  - *Inode:* key = `"INOD"` + inode id (**little-endian**, to spread inodes over FoundationDB nodes).
    Inode ids are 64-bit and monotonically increasing. Value: owner, permissions, times; for files:
    length, chunk size, chain range, shuffle seed; for directories: parent inode id and default
    layout (the parent id is needed to detect loops when moving a directory: check all ancestors of the
    destination); for symlinks: target path.
  - *Directory entry:* key = `"DENT"` + parent inode id + name; value = target inode id and type.
    Entries of a directory form a contiguous key range, so listing is a range query.
- **Transactions.** Read-only transactions for `fstat`, `lookup`, `listdir`. Read-write transactions
  for `create`, `link`, `unlink`, `rename`. FoundationDB tracks read/write key sets to detect
  conflicts; the meta service retries automatically. Several meta services can run in parallel while
  staying consistent (serializable snapshot isolation).
- **Dynamic attributes (length, open files).**
  - Read-only open fds are **not tracked** (training jobs open huge numbers of files at startup).
    A **session per write-opened fd** is kept; deleting a file with active write sessions is delayed
    until they close, and the meta service periodically checks client liveness and removes sessions of
    offline clients.
  - File length in the inode can lag. Clients report the maximum write position of each write-opened
    file **every 5 seconds**; if larger than the inode length and no concurrent truncate, it becomes
    the new length. That is only **eventually consistent**. On `close`/`fsync` the meta service gets
    the exact length by querying the last chunk's id and length from storage, which is costly because
    data is striped across chains.
  - To avoid transaction conflicts from several meta services updating the same file's length, length
    updates are assigned by **rendezvous hashing on inode id**.
  - Production stripe size is **200**. Small files touch far fewer chains, so the inode stores a hint
    of chains in use, starting at **16** and **doubling** as chunks are written to more chains,
    avoiding queries to all 200 chains. The same hint can speed up deleting small files.

### Data placement: chains and chain tables

Goal: highest bandwidth even during failures; throughput should scale **linearly with SSD count and
bisection bandwidth**; applications are locality-oblivious.

- **CRAQ chains.** Writes go to the head and propagate down the chain; reads can go to any target.
  Typically read traffic is spread evenly across all targets of a chain. Multiple targets are created
  per SSD and join different chains.
- **Example (6 nodes A-F, 1 SSD each, 5 targets per SSD = 30 targets, 3 replicas).** A naive chain
  table puts A1-B1-C1, D1-E1-F1, A2-B2-C2 and so on. Problem: if A fails, **its read traffic goes only
  to B and C**, saturating them, and replacing an SSD and re-syncing can take hours.
- **Balanced recovery.** Instead, pair each SSD with *every* other SSD across chains, so when A fails,
  each of the other five SSDs picks up **1/5 of A's reads**. The authors formulate chain-table
  construction as a **balanced incomplete block design** and solve it with an **integer programming
  solver**.
- **Chain version numbers.** A chain's version increments when it changes (for example a target goes
  offline). **Only the primary cluster manager changes chain tables.** Multiple chain tables can serve
  different needs, for example one for batch jobs and one for online services on disjoint nodes/SSDs.
  A chain can appear in multiple chain tables, and the table exists so the meta service can pick one
  per file and stripe over its chains.

### Write path (CRAQ in 3FS)

For a write arriving at a storage service (from a client or the predecessor in the chain):

1. Check the request's **chain version** against the latest known; reject if mismatched.
2. Use **RDMA Read** to pull the write data from the sender; if the sender died the read times out
   and the write aborts.
3. Take the **chunk lock** from a lock manager; concurrent writes to the same chunk are blocked and
   effectively serialized at the head.
4. Read the committed version into memory, apply the update, and store it as a **pending version**.
   A target may hold two versions: committed `v` and pending `u = v + 1`.
5. If this node is the **tail**, atomically replace committed with pending and acknowledge to the
   predecessor; otherwise forward the write to the successor. When committing, the current chain
   version is stored in the chunk metadata.
6. When an **ack** arrives back at a node, it commits its pending version, propagates the ack to its
   predecessor, and releases the chunk lock.

**Failure during a write (worked example from the notes).** Chain A to B to C. A forwards to B, B
fails instantly, and the forwarded write is lost. The cluster manager marks B offline, moves it to
the **end of the chain**, and broadcasts the new table. A gets the new table and forwards to C. C may
not have the new table yet and rejects; A **keeps forwarding** until C catches up and accepts.

### Read path

- If the target holds only a committed version, return it.
- Unlike classic CRAQ, **3FS does not ask the tail for the version**. If a chunk has both committed
  and pending versions, the service returns a **special status code**; the client either waits a short
  time and retries, or issues a **relaxed read** to get the pending version.

### Failure detection and the target state machine

- Heartbeats to the cluster manager act as **lease renewals**. The manager declares a service failed
  after `T` seconds without a heartbeat; a service that can't reach the manager for `T/2` stops
  serving and exits (so a partitioned service fences itself before the manager gives up on it).
- Meta services are stateless; the manager's list of online meta services is just service discovery.
- The manager keeps the global view of chain tables and target states. Each target has a
  **public state** (stored with chain tables, distributed to everyone) and a **local state** (known
  only to the storage service and the manager, in the manager's memory).

| Public state | Serves reads | Receives writes | Meaning |
|---|---|---|---|
| serving | yes | yes | alive and serving |
| syncing | no | yes | alive, data recovery in progress |
| waiting | no | no | alive, recovery not started |
| lastsrv | no | no | down, and was the last serving target |
| offline | no | no | down or storage medium failure |

| Local state | Meaning |
|---|---|
| up-to-date | alive and serving |
| online | alive; target is syncing or waiting |
| offline | service down or medium failure |

The manager periodically scans every chain and moves each target's public state according to a
transition table that takes (local state, current public state, predecessor's public state) as input.
Highlights: an offline target that was serving with no predecessor becomes **lastsrv** (so a
chain whose only up-to-date copy is down is not silently resumed from stale data); one with a
predecessor becomes offline; a returning target goes to **waiting**, then **syncing** only when its
predecessor is serving, then **serving** once it reports up-to-date. Other rules: bump the chain
version on any change; move offline targets to the chain's end; a service that finds any of its
targets in `lastsrv` or `offline` public state **exits at once** (it may be network-partitioned).

### Data recovery

Targets recover independently, overlapping with normal traffic.

**A returning storage service:** (1) pulls chain tables but does **not send heartbeats until all its
targets are marked offline** in the latest tables, forcing every target through recovery; (2) writes
that arrive during recovery are always **full-chunk-replace**, replacing the committed version and
dropping any pending one; (3) its predecessor first sends **dump-chunkmeta** to get chunk ids, chain
versions and committed/pending version numbers; (4) on **sync-done** it sets local state up-to-date
in its next heartbeat.

**The predecessor** (when it sees the successor online): starts forwarding normal writes to it as
full-chunk-replace writes (even if the client updated only part of a chunk), exchanges chunk
metadata, then transfers the chunks that need it, locking each chunk, reading its chain version,
committed version and content, sending a full-chunk-replace, then unlocking. Rules for what to send:

- only on the local target: transfer;
- only on the remote target: remove;
- local chain version greater than remote: transfer;
- chain versions equal but local committed version != remote pending version: transfer;
- otherwise the replicas are the same or being written by in-flight writes.

When done it sends **sync-done**.

### Chunk engine (on each SSD)

- Persistent state: a fixed number of **data files** for chunk contents plus a **RocksDB** instance
  for chunk metadata and system info; an in-memory metadata cache and a fast chunk allocator.
- Operations: `open/close` (load metadata from RocksDB, rebuild allocator), `get` (O(1) average
  hash-map lookup returning metadata and a reference-counted handle), `update` (**copy-on-write**:
  allocate a new chunk before modifying; old chunk stays readable until handles are released),
  `commit` (atomic RocksDB write batch and synchronous cache refresh).
- **Physical blocks:** 11 sizes from **64 KiB to 64 MiB** in powers of two. The allocator picks the
  size closest to the chunk size. Each size has a pool of **256 physical files**; block usage is kept
  in in-memory bitmaps. Freed blocks stay reserved and are reused first; when none remain,
  `fallocate()` creates 256 new contiguous blocks, reducing fragmentation.
- **Write on a chunk:** allocate a new block, read old data into a buffer, apply the update, write
  to the new block, then atomically update metadata and block statuses in RocksDB. **Appends** are
  optimized to write in place at the end of the existing block.

## Checkpointing on 3FS (paper Section VII-A)

The checkpoint manager exists to make interrupt-and-resume cheap, since LLM training runs for months
and failures are unavoidable.

- Parameters and optimizer states are split into chunks and written with the **3FS batch write API**,
  achieving **over 10 GiB/s per node**, so saving "completes in just a few seconds".
- State is first **asynchronously copied from GPU to CPU host memory**, then saved periodically,
  typically **every 5 minutes**.
- Each tensor's index and its offset within the checkpoint are recorded, so loading can locate any
  tensor; the **batch read API** loads in a few seconds.
- Result: after a failure, **only about the last 5 minutes of progress is lost**. For a cluster
  with thousands of nodes the authors call this overhead minimal.

(The paper does not describe the commit/atomicity protocol, retention policy or validation of
checkpoints. Atomic directory move from the 3FS design notes is the natural primitive for it.)

## Stability, validator and failure data (Section VII)

- **Validator:** runs weekly on every node. Checks frequencies and link status, CPU stress and
  memory bandwidth, a byte-by-byte **GPU memory test**, GEMM at full GPU memory (to catch faulty
  logic), intra-node all-reduce (NVLink bandwidth), and a **storage bandwidth stress test**. Faulty
  nodes are removed from scheduling.
- **Silent data corruption:** they saw compute errors and GPU memory errors **not caught by ECC**
  causing gradient-norm spikes, loss explosions and non-convergence.
- **GPU Xid errors over a year:** NVLink errors (Xid 74) are **42.57%** of GPU failures (mostly the
  NVLink Bridge connectors); illegal memory access (Xid 43) is **33.48%**; GPU memory ECC errors
  (Xid 63/64/94/95) about **2%**. Compared with another paper reporting NVLink-related failures at
  about 52% of failures. GPU ECC faults far outnumber CPU memory faults.
- **Network flash cuts:** InfiniBand link failures are about **30% of hardware faults excluding Xid 74**
  and occur at random times. Because most jobs span many nodes, one flaky link can fail a job.

## Future work in the paper

Next-generation PCIe nodes for MoE training (all-to-all heavy) with a **1:1 GPU-to-NIC ratio**; a
**multi-plane** network; considering RoCE instead of IB; a 128-port 400 Gbps RoCE switch in a
4-plane two-layer fat-tree could support up to **32,768 GPUs**.

## Limitations, critique and open questions

Stated by the authors:
- Request-to-send raises I/O latency in exchange for sustainable throughput.
- File length is only eventually consistent while a file is being written; exact length on
  `close`/`fsync` is expensive when data is striped over many chains.
- FUSE is slow for small random reads; fixing it with a native API means applications must adopt it.
- Disabling DCQCN and adaptive routing works because of their specific traffic mix and static
  placement; the paper does not claim it generalizes.

My own observations (not claims from the paper):
- The paper has **no latency distributions, IOPS, small-file metadata benchmarks, or failure-recovery
  timings** for 3FS. The 8 TB/s figure is aggregate read throughput on 180 nodes; capacity and
  write numbers are not given beyond 10 GiB/s per node for checkpoints.
- Replication is mirroring (the design notes' example uses 3 replicas per chain), which costs
  capacity; the paper and notes describe no erasure coding.
- It is a **training-optimized, single-tenant-ish** system: locality-oblivious, one trust domain,
  relying on a custom-tuned IB fabric. Multi-tenant concerns (isolation, quotas, auth) that matter for
  a cloud like CoreWeave are not addressed.
- Nothing is said about snapshots or tiering to cheaper storage; capacity is all-flash.
- The design notes call their read path "CRAQ" but it differs: it returns a status code when a
  pending version exists instead of asking the tail, so a read hitting a chunk under write may need a retry.

## Takeaways for interviews and design work

1. **Co-design storage with the network.** One fat-tree for compute and storage works if you isolate
   traffic classes (SL/VL), use static routing for incast-heavy storage traffic, and control incast
   at the receiver (request-to-send).
2. **Avoid the metadata bottleneck.** Put metadata in a transactional KV store with stateless
   metadata servers; hand the client the layout once so it computes chunk locations itself.
3. **Chunk id = inode id + chunk index, striped across chains.** Randomized, balanced placement
   spreads load; read from **any** replica (CRAQ) to use all SSD bandwidth.
4. **Design the chain table for recovery,** not just steady state. Pair each SSD with many others
   (balanced incomplete block design) so a failure spreads its read load across the cluster.
5. **Fence failed nodes with leases:** a service stops after `T/2` of silence while the manager waits
   `T`.
6. **A `lastsrv` state** prevents resuming from stale replicas when the last up-to-date copy is down.
7. **Bypass FUSE for the hot path** with an `io_uring`-style shared-memory API (Iov for buffers, Ior
   for the ring) and batch small reads into fewer RPCs.
8. **Checkpoint cheaply and often:** snapshot GPU to CPU memory asynchronously, write sharded with a
   batch API at over 10 GiB/s per node, and checkpoint every 5 minutes so a failure costs minutes.

### Short answers to likely questions

- *Why CRAQ and not Raft/Paxos for data?* Reads are the dominant workload and CRAQ lets every
  replica serve reads, using all SSDs. Write-all means writes need every replica, so membership
  changes and recovery are handled by the cluster manager and chain versions.
- *What happens when a chain member dies mid-write?* The manager marks it offline, moves it to the
  end, bumps the chain version, and predecessors resend to the new successor; stale-version
  requests are rejected until nodes learn the new table.
- *How does it recover a replaced SSD without stalling the cluster?* The returning target runs the
  sync protocol with its predecessor (dump-chunkmeta, compare, full-chunk-replace copy) while
  taking normal writes as full-chunk replaces; reads are not routed to it until it is `serving`.
- *Why not a kernel client?* Hard to debug and upgrade; they chose a user-space native client with
  an `io_uring`-like interface.
- *Why a file system rather than an object store?* Atomic rename of directories, recursive delete,
  hard links and symlinks for snapshots, and drop-in compatibility with file-based data loaders.
- *How does this relate to CoreWeave CAIOS/LOTA?* (My comparison.) CAIOS keeps an S3-compatible
  origin store and adds a per-GPU-node NVMe cache (LOTA), so reads become local after warm-up.
  3FS makes the SSD fleet itself the primary store and has every read hit remote SSDs over RDMA, with
  no client data cache. 3FS wins on POSIX semantics and cold random-read throughput; the cache
  approach wins on cheap capacity (object storage origin) and repeated epochs.

## Related notes

- [Mooncake (arXiv 2024)](mooncake-arxiv-2024.md) and [LMCache (arXiv 2025)](lmcache-arxiv-2025.md):
  KV-cache serving; LMCache's paper lists 3FS among KV storage layers
- [vLLM (SOSP 2023)](vllm-sosp-2023.md)
- [FoundationDB Record Layer (SIGMOD 2019)](foundationdb-record-layer-sigmod-2019.md): background on
  FoundationDB, which 3FS uses for metadata
