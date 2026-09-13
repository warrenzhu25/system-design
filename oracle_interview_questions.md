# Oracle Interview Questions

---

## Contents

**Coding**
1. [Set Matrix Zeroes (In Place)](#1-set-matrix-zeroes-in-place)

**System Design**
2. [Online Presence System](#2-system-design--online-presence-system)

**Behavioral**
3. [Behavioral Themes](#3-behavioral-themes)

---

## 1. Set Matrix Zeroes (In Place)

**Problem Statement:**
Given an `m x n` integer matrix, if an element is `0`, set its entire row and column to `0`. Do it
in place, using `O(1)` extra space (i.e., don't build a separate set of rows/columns to zero out).

This is the one fully publicly-visible problem from Oracle's interview bank (most of the bank is
paywalled behind forum membership beyond title/tags: array, grid, hashmap, medium difficulty,
~30 minutes).

**Example:**
```
Input:
[[1,1,1],
 [1,0,1],
 [1,1,1]]

Output:
[[1,0,1],
 [0,0,0],
 [1,0,1]]
```

**Test Cases:**

| Input | Output |
|---|---|
| `[[1,1,1],[1,0,1],[1,1,1]]` | `[[1,0,1],[0,0,0],[1,0,1]]` |
| `[[0,1,2,0],[3,4,5,2],[1,3,1,5]]` | `[[0,0,0,0],[0,4,5,0],[0,3,1,0]]` |
| `[[1]]` | `[[1]]` |
| `[[0]]` | `[[0]]` |
| No zeroes present | matrix unchanged |

**Key Insights:**
1. The naive solution uses `O(m + n)` extra space (a set of rows and a set of columns to zero out
   afterward) — that's an easy warm-up; the interview bar is the `O(1)`-extra-space version.
2. Use the matrix's own first row and first column as the marker arrays: if `matrix[r][c] == 0`, mark
   `matrix[r][0] = 0` and `matrix[0][c] = 0` instead of writing to a separate structure.
3. Because the first row and first column are now doing double duty (both data and markers), record
   *before* mutating anything whether the first row and first column originally contained a zero —
   otherwise you can't tell at the end whether to zero them out too.
4. Process the rest of the matrix (rows/cols `1..end`) using the markers first, then handle row 0 and
   column 0 last, using the two booleans captured in step 3.

**Python Solution:**
```python
def set_zeroes(matrix: list[list[int]]) -> None:
    """
    Mutates matrix in place.
    Time:  O(m * n)
    Space: O(1) extra (beyond the input matrix itself)
    """
    if not matrix or not matrix[0]:
        return

    m, n = len(matrix), len(matrix[0])
    first_row_has_zero = any(matrix[0][c] == 0 for c in range(n))
    first_col_has_zero = any(matrix[r][0] == 0 for r in range(m))

    for r in range(1, m):
        for c in range(1, n):
            if matrix[r][c] == 0:
                matrix[r][0] = 0
                matrix[0][c] = 0

    for r in range(1, m):
        for c in range(1, n):
            if matrix[r][0] == 0 or matrix[0][c] == 0:
                matrix[r][c] = 0

    if first_row_has_zero:
        for c in range(n):
            matrix[0][c] = 0
    if first_col_has_zero:
        for r in range(m):
            matrix[r][0] = 0
```

**Follow-Up Questions:**
1. What if the input is ragged (rows of different lengths)? → guard every inner loop with
   `c < len(matrix[r])` instead of assuming a fixed `n`; the first-row/first-column marker trick still
   works as long as row 0 itself is treated as the longest row (or you fall back to explicit row/column
   sets for a ragged matrix, since a single shared `n` no longer makes sense).
2. Generalize to a matrix of characters where a designated "zero" value isn't literally `0` → parameterize
   the sentinel value the function checks for; the algorithm is otherwise unchanged.
3. Can you do it in a single pass instead of three? → not while keeping `O(1)` extra space: you need the
   first pass to detect zeroes before their positions get overwritten by the marker-writing pass.

---

## 2. System Design — Online Presence System

**Problem Statement:**
Design an online-presence system, like a chat app's "online / away / offline" indicator for each user,
that scales to hundreds of millions of users with near-real-time updates.

**Functional Requirements:**
- Track each user's current presence state (`online`, `away`, `offline`).
- Let a client query the presence of its contacts/subscribed users.
- Propagate a presence change to interested subscribers promptly.
- Presence should reflect ungraceful disconnects (app killed, network drop) without requiring an
  explicit "going offline" signal from the client.

**Non-Functional Requirements:**
- Near-real-time propagation (seconds, not minutes) to subscribers of a presence change.
- Scale to hundreds of millions of concurrent users without a single bottleneck component.
- Tolerate a spike of simultaneous disconnects (e.g., a mobile network outage) without cascading load.

**High-Level Design:**
1. **Heartbeat + TTL**: each connected client periodically pings a lightweight heartbeat endpoint (or the
   heartbeat rides on an existing persistent connection, e.g. a WebSocket). Presence state is written to
   a fast key-value store (e.g., Redis) with a TTL slightly longer than the heartbeat interval.
2. **Implicit offline detection**: because the TTL entry naturally expires if heartbeats stop, an
   ungraceful disconnect resolves to "offline" for free — there's no need for the server to detect the
   disconnect explicitly or run a separate reaper process for the common case.
3. **Fan-out on change**: when a user's presence changes (a new heartbeat after being offline, or a TTL
   expiry), publish a presence-changed event on a per-user pub/sub channel; only that user's actual
   subscribers (contacts, or viewers of their profile) receive it — never a global broadcast.
4. **Read path**: a client fetches a contact's presence either by subscribing to that pub/sub channel
   (for a live-updating UI) or via a direct point read of the KV store (for a one-off check, e.g.
   rendering a contact list on app open).
5. **High-fanout accounts**: a small number of accounts (e.g., a public/celebrity account) can have
   millions of subscribers watching one presence value — cache the current value aggressively and
   consider coalescing/delaying updates for very high-fanout accounts rather than pushing every flicker
   in real time to every subscriber.

**Data Model (sketch):**
```
presence(user_id, status, last_heartbeat_ts)   # KV store, TTL ~= 2x heartbeat interval
subscriptions(user_id -> set of subscriber_ids)  # or derived from an existing contacts/social graph
```

**Scaling & Reliability:**
- Shard the KV store by `user_id` (consistent hashing) so heartbeat writes spread evenly across nodes.
- Heartbeat traffic is the dominant write volume at this scale — keep the heartbeat payload minimal and
  the write path a single fast key update, not a multi-step transaction.
- Pub/sub fan-out is per-subscriber-count, not global, so a spike in disconnects (e.g. a network outage
  recovering) causes a burst of presence-changed events bounded by how many users are actually affected,
  not the whole user base.
- For very high-fanout accounts, decouple "true" presence (as tracked internally) from "displayed"
  presence (which can lag slightly or be sampled) to cap fan-out cost.

**Follow-Up Questions:**
1. How do you distinguish "away" from "offline" if both look like "no recent activity"? → "away" is
   typically a client-reported state (app backgrounded but still holding a connection) sent explicitly,
   distinct from the TTL-driven "offline" which requires no client cooperation at all.
2. A user has a flaky connection and flaps between online/offline every few seconds — how do you avoid
   spamming subscribers? → debounce presence-changed events (only publish if the state has been stable
   for some minimum duration) rather than publishing on every raw heartbeat gap.
3. How would you test TTL-based expiry behavior deterministically? → inject a fake clock into the
   heartbeat/TTL logic in tests rather than relying on wall-clock sleeps.

---

## 3. Behavioral Themes

See [`behavioral_interview.md`](./behavioral_interview.md) for general STAR-method prep. Themes
specific to Oracle's loop:

- **Ownership under ambiguity**: standard enterprise-software behavioral theme — can you drive a project
  to completion with unclear requirements, and take ownership of outcomes (including failures) rather
  than deflecting.
- **Proactive requirement clarification**: a story showing you clarified ambiguous requirements up front
  (rather than building the wrong thing and finding out later) tends to land well here.
- **Leadership without authority**: influencing a technical direction or unblocking a cross-team
  dependency without having formal reporting authority over the people involved.

---

## References

Sources used for compiling these questions:
- [Oracle Interview Questions - 1point3acres](https://www.1point3acres.com/interview/problems/company/oracle)

Note: the source page requires forum membership to view full question text/discussion threads for most
of the bank; the "Set Matrix Zeroes" coding problem above was one of the few fully publicly-visible
entries. The system-design and behavioral entries were reconstructed and expanded from the publicly
visible question titles/tags into complete, solvable problem statements with original test cases and
solutions.
