# Snowflake Interview Questions

---

## 1. Progressive Tax Calculation

**Problem Statement:**
You are provided with a 0-indexed 2D integer array called `brackets`. Each item in this list is a pair `[upper_bound, tax_rate]` representing a tax bracket where:
- `upper_bound` is the highest income amount included in this bracket
- `tax_rate` is the percentage of tax charged for this bracket

The list `brackets` is sorted by `upper_bound` from smallest to largest. All upper bounds are distinct.

Given an integer `income`, calculate the total tax based on a progressive tax system.

**How Progressive Tax Works:**
Tax is calculated in steps or "chunks":
1. Money up to the first bracket's limit is taxed at the first rate
2. Any money above the first limit—but below the second limit—is taxed at the second rate
3. This pattern continues for each subsequent bracket until all your income is accounted for

Return the total amount of tax paid.

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `brackets = [[3,50],[7,10],[12,25]], income = 10` | 2.65 | First $3 at 50% = $1.50, next $4 (3-7) at 10% = $0.40, next $3 (7-10) at 25% = $0.75. Total = $2.65 |
| `brackets = [[1,0],[4,25],[5,50]], income = 2` | 0.25 | First $1 at 0% = $0, next $1 (1-2) at 25% = $0.25. Total = $0.25 |

**Constraints:**
- `1 <= brackets.length <= 100`
- `1 <= upper_bound <= 1000`
- `0 <= tax_rate <= 100`
- `0 <= income <= 1000`
- `upper_bound` increases strictly (the list is sorted and values are unique)

**Key Insight:**
For each bracket, calculate the taxable amount within that bracket by finding how much income falls between the previous upper bound and the current upper bound. Multiply by the tax rate and accumulate.

**Python Solution:**
```python
def calculateTax(brackets: list[list[int]], income: int) -> float:
    """
    Calculate total tax using progressive tax brackets.

    Time: O(n) where n is number of brackets
    Space: O(1)
    """
    total_tax = 0.0
    prev_upper = 0

    for upper_bound, tax_rate in brackets:
        if income <= prev_upper:
            break

        # Calculate taxable amount in this bracket
        taxable = min(income, upper_bound) - prev_upper

        # Add tax for this bracket (rate is percentage, so divide by 100)
        total_tax += taxable * (tax_rate / 100)

        prev_upper = upper_bound

    return total_tax
```

**Java Solution:**
```java
public double calculateTax(int[][] brackets, int income) {
    double totalTax = 0.0;
    int prevUpper = 0;

    for (int[] bracket : brackets) {
        int upperBound = bracket[0];
        int taxRate = bracket[1];

        if (income <= prevUpper) {
            break;
        }

        // Calculate taxable amount in this bracket
        int taxable = Math.min(income, upperBound) - prevUpper;

        // Add tax for this bracket
        totalTax += taxable * (taxRate / 100.0);

        prevUpper = upperBound;
    }

    return totalTax;
}
```

**TypeScript Solution:**
```typescript
function calculateTax(brackets: number[][], income: number): number {
    let totalTax = 0;
    let prevUpper = 0;

    for (const [upperBound, taxRate] of brackets) {
        if (income <= prevUpper) {
            break;
        }

        const taxable = Math.min(income, upperBound) - prevUpper;
        totalTax += taxable * (taxRate / 100);

        prevUpper = upperBound;
    }

    return totalTax;
}
```

**Walkthrough - Example 1:**
```
brackets = [[3,50],[7,10],[12,25]], income = 10

Bracket 1: [3, 50]
  - prevUpper = 0, upperBound = 3
  - taxable = min(10, 3) - 0 = 3
  - tax = 3 * 0.50 = 1.50
  - prevUpper = 3

Bracket 2: [7, 10]
  - prevUpper = 3, upperBound = 7
  - taxable = min(10, 7) - 3 = 4
  - tax = 4 * 0.10 = 0.40
  - prevUpper = 7

Bracket 3: [12, 25]
  - prevUpper = 7, upperBound = 12
  - taxable = min(10, 12) - 7 = 3
  - tax = 3 * 0.25 = 0.75
  - prevUpper = 12

Total tax = 1.50 + 0.40 + 0.75 = 2.65
```

**Complexity Analysis:**

| Aspect | Complexity |
|--------|------------|
| Time | O(n) - single pass through brackets |
| Space | O(1) - only tracking running totals |

**Note:** LeetCode problem #2303 - Calculate Amount Paid in Taxes

---

## 2. Tree Levels After Node Deletions

**Problem Statement:**
You are given the root of a binary tree where every node has a unique ID. You are also given a list called `toDelete` containing the IDs of nodes to remove.

**Deletion Rules:**
1. When a node is deleted, it is removed from the tree
2. Its children move up to replace it, attaching to the parent of the deleted node
3. If multiple nodes in a row are deleted, descendants keep moving up until they find a non-deleted ancestor
4. If no non-deleted ancestor exists, those descendants become new roots

After applying all deletions, you may end up with a forest (collection of separate trees). Find the **maximum depth** (number of levels) in this final forest.

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `root = [1,2,3,4,5,null,6], toDelete = [2]` | 3 | Delete node 2. Children 4 and 5 move up. Longest path is 1 -> 3 -> 6 (3 levels) |
| `root = [1,2,3,4,null,null,5], toDelete = [1,3]` | 2 | Delete nodes 1 and 3. Node 2 (with child 4) and node 5 become roots. Max depth is 2 |

**Constraints:**
- `1 <= number of nodes <= 10^5`
- `-10^5 <= Node.val <= 10^5`
- All `Node.val` entries are unique
- `0 <= toDelete.length <= number of nodes`
- All IDs in `toDelete` are valid node IDs

**Key Insight:**
When a node is deleted, its children's effective depth decreases. We need to track the "effective depth" of each node accounting for deleted ancestors. Use DFS and track how many non-deleted ancestors each node has.

**Python Solution:**
```python
from typing import Optional

class TreeNode:
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def maxDepthAfterDeletions(root: Optional[TreeNode], toDelete: list[int]) -> int:
    """
    Find max depth of forest after deleting specified nodes.

    Key idea: Track effective depth (counting only non-deleted ancestors).
    When we delete a node, its children's effective depth doesn't increase
    from their grandparent's perspective.

    Time: O(n)
    Space: O(n) for recursion stack and delete set
    """
    if not root:
        return 0

    delete_set = set(toDelete)
    max_depth = 0

    def dfs(node: Optional[TreeNode], effective_depth: int) -> None:
        nonlocal max_depth

        if not node:
            return

        is_deleted = node.val in delete_set

        if is_deleted:
            # Node is deleted - children start fresh or continue from grandparent
            # effective_depth stays same (children take this node's place)
            dfs(node.left, effective_depth)
            dfs(node.right, effective_depth)
        else:
            # Node is kept - increment depth
            new_depth = effective_depth + 1
            max_depth = max(max_depth, new_depth)
            dfs(node.left, new_depth)
            dfs(node.right, new_depth)

    dfs(root, 0)
    return max_depth
```

**Java Solution:**
```java
class TreeNode {
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int val) { this.val = val; }
}

class Solution {
    private Set<Integer> deleteSet;
    private int maxDepth;

    public int maxDepthAfterDeletions(TreeNode root, int[] toDelete) {
        if (root == null) return 0;

        deleteSet = new HashSet<>();
        for (int val : toDelete) {
            deleteSet.add(val);
        }

        maxDepth = 0;
        dfs(root, 0);
        return maxDepth;
    }

    private void dfs(TreeNode node, int effectiveDepth) {
        if (node == null) return;

        boolean isDeleted = deleteSet.contains(node.val);

        if (isDeleted) {
            // Children move up to take this node's place
            dfs(node.left, effectiveDepth);
            dfs(node.right, effectiveDepth);
        } else {
            // Node is kept - increment depth
            int newDepth = effectiveDepth + 1;
            maxDepth = Math.max(maxDepth, newDepth);
            dfs(node.left, newDepth);
            dfs(node.right, newDepth);
        }
    }
}
```

**TypeScript Solution:**
```typescript
class TreeNode {
    val: number;
    left: TreeNode | null;
    right: TreeNode | null;
    constructor(val: number = 0, left: TreeNode | null = null, right: TreeNode | null = null) {
        this.val = val;
        this.left = left;
        this.right = right;
    }
}

function maxDepthAfterDeletions(root: TreeNode | null, toDelete: number[]): number {
    if (!root) return 0;

    const deleteSet = new Set(toDelete);
    let maxDepth = 0;

    function dfs(node: TreeNode | null, effectiveDepth: number): void {
        if (!node) return;

        const isDeleted = deleteSet.has(node.val);

        if (isDeleted) {
            // Children move up to take this node's place
            dfs(node.left, effectiveDepth);
            dfs(node.right, effectiveDepth);
        } else {
            // Node is kept - increment depth
            const newDepth = effectiveDepth + 1;
            maxDepth = Math.max(maxDepth, newDepth);
            dfs(node.left, newDepth);
            dfs(node.right, newDepth);
        }
    }

    dfs(root, 0);
    return maxDepth;
}
```

**Walkthrough - Example 1:**
```
Tree:       1
           / \
          2   3
         / \   \
        4   5   6

toDelete = [2]

DFS traversal:
- Node 1: not deleted, effectiveDepth = 1, maxDepth = 1
  - Node 2: DELETED, effectiveDepth stays 1
    - Node 4: not deleted, effectiveDepth = 2, maxDepth = 2
    - Node 5: not deleted, effectiveDepth = 2, maxDepth = 2
  - Node 3: not deleted, effectiveDepth = 2, maxDepth = 2
    - Node 6: not deleted, effectiveDepth = 3, maxDepth = 3

Result: 3 (path 1 -> 3 -> 6)
```

**Walkthrough - Example 2:**
```
Tree:       1
           / \
          2   3
         /     \
        4       5

toDelete = [1, 3]

DFS traversal:
- Node 1: DELETED, effectiveDepth stays 0
  - Node 2: not deleted, effectiveDepth = 1, maxDepth = 1
    - Node 4: not deleted, effectiveDepth = 2, maxDepth = 2
  - Node 3: DELETED, effectiveDepth stays 0
    - Node 5: not deleted, effectiveDepth = 1, maxDepth = 2

Result: 2 (path 2 -> 4)
```

**Complexity Analysis:**

| Aspect | Complexity |
|--------|------------|
| Time | O(n) - visit each node once |
| Space | O(n) - recursion stack + hash set |

---

## 3. Happy Number

**Problem Statement:**
Determine if a number `n` is "happy."

**How It Works:**
1. Start with any positive number
2. Square each digit and add the results together
3. Replace the original number with this new sum
4. Repeat until either:
   - You reach **1** → the number is **happy** (return `true`)
   - You enter a **cycle** that never reaches 1 → **not happy** (return `false`)

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `n = 19` | `true` | 1² + 9² = 82 → 8² + 2² = 68 → 6² + 8² = 100 → 1² + 0² + 0² = 1 |
| `n = 2` | `false` | Loops endlessly without reaching 1 |

**Constraints:**
- `1 <= n <= 2^31 - 1`

**Key Insight:**
The sequence will either reach 1 or enter a cycle. Use a HashSet to detect cycles, or use Floyd's cycle detection (fast/slow pointers) for O(1) space.

**Python Solution:**
```python
def isHappy(n: int) -> bool:
    """
    Check if n is a happy number using HashSet for cycle detection.

    Time: O(log n) - sum of squares reduces large numbers quickly
    Space: O(log n) - storing seen numbers
    """
    def get_sum_of_squares(num: int) -> int:
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    seen = set()

    while n != 1 and n not in seen:
        seen.add(n)
        n = get_sum_of_squares(n)

    return n == 1


def isHappyFloyd(n: int) -> bool:
    """
    O(1) space solution using Floyd's cycle detection.
    """
    def get_sum_of_squares(num: int) -> int:
        total = 0
        while num > 0:
            digit = num % 10
            total += digit * digit
            num //= 10
        return total

    slow = n
    fast = get_sum_of_squares(n)

    while fast != 1 and slow != fast:
        slow = get_sum_of_squares(slow)
        fast = get_sum_of_squares(get_sum_of_squares(fast))

    return fast == 1
```

**Java Solution:**
```java
class Solution {
    public boolean isHappy(int n) {
        Set<Integer> seen = new HashSet<>();

        while (n != 1 && !seen.contains(n)) {
            seen.add(n);
            n = getSumOfSquares(n);
        }

        return n == 1;
    }

    private int getSumOfSquares(int n) {
        int sum = 0;
        while (n > 0) {
            int digit = n % 10;
            sum += digit * digit;
            n /= 10;
        }
        return sum;
    }

    // O(1) space using Floyd's cycle detection
    public boolean isHappyFloyd(int n) {
        int slow = n;
        int fast = getSumOfSquares(n);

        while (fast != 1 && slow != fast) {
            slow = getSumOfSquares(slow);
            fast = getSumOfSquares(getSumOfSquares(fast));
        }

        return fast == 1;
    }
}
```

**TypeScript Solution:**
```typescript
function isHappy(n: number): boolean {
    const getSumOfSquares = (num: number): number => {
        let sum = 0;
        while (num > 0) {
            const digit = num % 10;
            sum += digit * digit;
            num = Math.floor(num / 10);
        }
        return sum;
    };

    const seen = new Set<number>();

    while (n !== 1 && !seen.has(n)) {
        seen.add(n);
        n = getSumOfSquares(n);
    }

    return n === 1;
}

// O(1) space version
function isHappyFloyd(n: number): boolean {
    const getSumOfSquares = (num: number): number => {
        let sum = 0;
        while (num > 0) {
            const digit = num % 10;
            sum += digit * digit;
            num = Math.floor(num / 10);
        }
        return sum;
    };

    let slow = n;
    let fast = getSumOfSquares(n);

    while (fast !== 1 && slow !== fast) {
        slow = getSumOfSquares(slow);
        fast = getSumOfSquares(getSumOfSquares(fast));
    }

    return fast === 1;
}
```

**Walkthrough - n = 19:**
```
Step 1: 1² + 9² = 1 + 81 = 82
Step 2: 8² + 2² = 64 + 4 = 68
Step 3: 6² + 8² = 36 + 64 = 100
Step 4: 1² + 0² + 0² = 1

Reached 1 → Happy! ✓
```

**Why Floyd's Algorithm Works:**
- If there's a cycle, fast pointer (moving 2 steps) will eventually meet slow pointer (moving 1 step)
- If sequence reaches 1, fast pointer gets stuck at 1 (since 1² = 1)
- No extra memory needed for tracking seen numbers

**Complexity Analysis:**

| Approach | Time | Space |
|----------|------|-------|
| HashSet | O(log n) | O(log n) |
| Floyd's Cycle Detection | O(log n) | O(1) |

**Note:** LeetCode problem #202 - Happy Number

---

## 4. SnowCal Language Interpreter

**Problem Statement:**
SnowCal is a simple programming language that stores exactly one integer `X` in memory, starting at 0.

**Available Commands:**
| Command | Description |
|---------|-------------|
| `ADD Y` | Add value Y to X |
| `MUL Y` | Multiply X by value Y |
| `FUN F` | Start defining a function named F |
| `END` | Finish defining the current function |
| `INV F` | Call (invoke) function F |

**Language Rules:**
- Every function has a unique name
- No nested function definitions allowed
- Functions can be defined but never used
- If `INV F` appears, function F is guaranteed to exist
- Commands inside `FUN ... END` only execute when invoked via `INV`

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `["MUL 2", "ADD 3"]` | 3 | X=0 → 0*2=0 → 0+3=3 |
| `["FUN INCREMENT", "ADD 1", "END", "INV INCREMENT", "MUL 2", "ADD 3"]` | 5 | Define func, invoke (X=1), *2=2, +3=5 |
| `["FUN INCREMENT", "ADD 1", "END", "FUN INCREMENT2", "ADD 1", "MUL 2", "END", "MUL 2", "INV INCREMENT2", "ADD 3", "INV INCREMENT"]` | 6 | See walkthrough below |

**Constraints:**
- `1 <= program.length <= 10,000`
- `-10^9 <= Y <= 10^9`
- No nested functions

**Key Insight:**
Two-pass approach: First pass collects function definitions (store the commands between FUN and END). Second pass executes the program, invoking stored functions when needed.

**Python Solution:**
```python
def interpret(program: list[str]) -> int:
    """
    Interpret SnowCal program and return final value of X.

    Time: O(n * m) where n is program length, m is max function calls
    Space: O(n) for storing function definitions
    """
    functions = {}  # function_name -> list of commands
    x = 0

    # First pass: collect function definitions
    i = 0
    while i < len(program):
        line = program[i]

        if line.startswith("FUN "):
            func_name = line[4:]  # Extract function name
            func_body = []
            i += 1

            # Collect commands until END
            while i < len(program) and program[i] != "END":
                func_body.append(program[i])
                i += 1

            functions[func_name] = func_body
        i += 1

    # Helper to execute a single command
    def execute(cmd: str) -> None:
        nonlocal x

        if cmd.startswith("ADD "):
            x += int(cmd[4:])
        elif cmd.startswith("MUL "):
            x *= int(cmd[4:])
        elif cmd.startswith("INV "):
            func_name = cmd[4:]
            for func_cmd in functions[func_name]:
                execute(func_cmd)

    # Second pass: execute main program (skip function definitions)
    i = 0
    while i < len(program):
        line = program[i]

        if line.startswith("FUN "):
            # Skip function definition
            while i < len(program) and program[i] != "END":
                i += 1
        elif line != "END":
            execute(line)
        i += 1

    return x
```

**Java Solution:**
```java
import java.util.*;

class Solution {
    private Map<String, List<String>> functions;
    private long x;

    public long interpret(String[] program) {
        functions = new HashMap<>();
        x = 0;

        // First pass: collect function definitions
        int i = 0;
        while (i < program.length) {
            String line = program[i];

            if (line.startsWith("FUN ")) {
                String funcName = line.substring(4);
                List<String> funcBody = new ArrayList<>();
                i++;

                while (i < program.length && !program[i].equals("END")) {
                    funcBody.add(program[i]);
                    i++;
                }

                functions.put(funcName, funcBody);
            }
            i++;
        }

        // Second pass: execute main program
        i = 0;
        while (i < program.length) {
            String line = program[i];

            if (line.startsWith("FUN ")) {
                // Skip function definition
                while (i < program.length && !program[i].equals("END")) {
                    i++;
                }
            } else if (!line.equals("END")) {
                execute(line);
            }
            i++;
        }

        return x;
    }

    private void execute(String cmd) {
        if (cmd.startsWith("ADD ")) {
            x += Long.parseLong(cmd.substring(4));
        } else if (cmd.startsWith("MUL ")) {
            x *= Long.parseLong(cmd.substring(4));
        } else if (cmd.startsWith("INV ")) {
            String funcName = cmd.substring(4);
            for (String funcCmd : functions.get(funcName)) {
                execute(funcCmd);
            }
        }
    }
}
```

**TypeScript Solution:**
```typescript
function interpret(program: string[]): number {
    const functions: Map<string, string[]> = new Map();
    let x = 0;

    // First pass: collect function definitions
    let i = 0;
    while (i < program.length) {
        const line = program[i];

        if (line.startsWith("FUN ")) {
            const funcName = line.substring(4);
            const funcBody: string[] = [];
            i++;

            while (i < program.length && program[i] !== "END") {
                funcBody.push(program[i]);
                i++;
            }

            functions.set(funcName, funcBody);
        }
        i++;
    }

    // Helper to execute a single command
    const execute = (cmd: string): void => {
        if (cmd.startsWith("ADD ")) {
            x += parseInt(cmd.substring(4));
        } else if (cmd.startsWith("MUL ")) {
            x *= parseInt(cmd.substring(4));
        } else if (cmd.startsWith("INV ")) {
            const funcName = cmd.substring(4);
            for (const funcCmd of functions.get(funcName)!) {
                execute(funcCmd);
            }
        }
    };

    // Second pass: execute main program
    i = 0;
    while (i < program.length) {
        const line = program[i];

        if (line.startsWith("FUN ")) {
            while (i < program.length && program[i] !== "END") {
                i++;
            }
        } else if (line !== "END") {
            execute(line);
        }
        i++;
    }

    return x;
}
```

**Walkthrough - Case 3:**
```
Program:
["FUN INCREMENT", "ADD 1", "END",
 "FUN INCREMENT2", "ADD 1", "MUL 2", "END",
 "MUL 2", "INV INCREMENT2", "ADD 3", "INV INCREMENT"]

Pass 1 - Collect functions:
  INCREMENT  → ["ADD 1"]
  INCREMENT2 → ["ADD 1", "MUL 2"]

Pass 2 - Execute (X starts at 0):
  MUL 2         → X = 0 * 2 = 0
  INV INCREMENT2:
    ADD 1       → X = 0 + 1 = 1
    MUL 2       → X = 1 * 2 = 2
  ADD 3         → X = 2 + 3 = 5
  INV INCREMENT:
    ADD 1       → X = 5 + 1 = 6

Result: 6
```

**Edge Cases:**
- Function defined but never called → ignored
- `MUL` when X=0 → stays 0
- Negative values for ADD/MUL
- Functions calling other functions (supported by recursive execute)

**Complexity Analysis:**

| Aspect | Complexity |
|--------|------------|
| Time | O(n × m) where n = program length, m = max invocation depth |
| Space | O(n) for storing function definitions |

---

## 5. Recipe Sequence Matcher

**Problem Statement:**
You have a main list of ingredients called `ingredients` and a list of `recipes`. Each recipe is a smaller list of ingredients. For every recipe, check if it exists inside the main ingredients list as a **contiguous sequence** (no gaps, exact order).

**Interview Structure:**
| Part | Requirement | Approach |
|------|-------------|----------|
| Part 1 | Fast queries, preprocessing allowed | Rolling Hash |
| Part 2 | O(1) extra space | Two Pointers |
| Part 3 | Streaming data (can't store all ingredients) | Aho-Corasick |

**Example:**
```python
ingredients = ["bun", "lettuce", "tomato", "patty", "cheese", "onion"]
recipes = [
    ["lettuce", "tomato", "patty"],  # True (contiguous in middle)
    ["tomato", "cheese"],            # False ("patty" is between them)
    ["patty", "cheese"],             # True (adjacent)
]
# Output: [True, False, True]
```

**Test Case:**
```python
ingredients = ["a", "b", "c", "d", "e"]
recipes = [["b", "c"], ["c", "e"], ["a"], ["d", "e", "f"]]
# Output: [True, False, True, False]
```

---

### Part 1: Hash-Based Solution (Fast Queries)

**Approach:** Precompute rolling hashes for all contiguous subsequences. Use double hashing to minimize collisions.

**Python Solution:**
```python
from collections import defaultdict

class RecipeMatcher:
    _MOD1 = 1_000_000_007
    _MOD2 = 1_000_000_009
    _BASE1 = 911_382_323
    _BASE2 = 972_663_749

    def __init__(self, ingredients: list[str]):
        self._token_id: dict[str, int] = {}
        self._next_id = 1
        self._hashes_by_len: dict[int, set[tuple[int, int]]] = defaultdict(set)

        # Assign IDs to all tokens
        for token in ingredients:
            self._get_token_id(token)

        # Precompute hashes for all contiguous subsequences
        n = len(ingredients)
        for i in range(n):
            h1, h2 = 0, 0
            for j in range(i, n):
                token_value = self._get_token_id(ingredients[j])
                h1 = (h1 * self._BASE1 + token_value) % self._MOD1
                h2 = (h2 * self._BASE2 + token_value) % self._MOD2
                self._hashes_by_len[j - i + 1].add((h1, h2))

    def _get_token_id(self, token: str) -> int:
        if token not in self._token_id:
            self._token_id[token] = self._next_id
            self._next_id += 1
        return self._token_id[token]

    def can_make(self, recipe: list[str]) -> bool:
        if not recipe:
            return True

        h1, h2 = 0, 0
        for token in recipe:
            token_value = self._token_id.get(token)
            if token_value is None:
                return False  # Unknown ingredient
            h1 = (h1 * self._BASE1 + token_value) % self._MOD1
            h2 = (h2 * self._BASE2 + token_value) % self._MOD2

        return (h1, h2) in self._hashes_by_len[len(recipe)]


def match_recipes(ingredients: list[str], recipes: list[list[str]]) -> list[bool]:
    matcher = RecipeMatcher(ingredients)
    return [matcher.can_make(recipe) for recipe in recipes]
```

**Java Solution:**
```java
import java.util.*;

class RecipeMatcher {
    private static final long MOD1 = 1_000_000_007L;
    private static final long MOD2 = 1_000_000_009L;
    private static final long BASE1 = 911_382_323L;
    private static final long BASE2 = 972_663_749L;

    private Map<String, Integer> tokenId = new HashMap<>();
    private int nextId = 1;
    private Map<Integer, Set<Long>> hashesByLen = new HashMap<>();

    public RecipeMatcher(String[] ingredients) {
        for (String token : ingredients) {
            getTokenId(token);
        }

        int n = ingredients.length;
        for (int i = 0; i < n; i++) {
            long h1 = 0, h2 = 0;
            for (int j = i; j < n; j++) {
                int tokenValue = getTokenId(ingredients[j]);
                h1 = (h1 * BASE1 + tokenValue) % MOD1;
                h2 = (h2 * BASE2 + tokenValue) % MOD2;
                int len = j - i + 1;
                hashesByLen.computeIfAbsent(len, k -> new HashSet<>())
                           .add(h1 * MOD2 + h2);
            }
        }
    }

    private int getTokenId(String token) {
        return tokenId.computeIfAbsent(token, k -> nextId++);
    }

    public boolean canMake(String[] recipe) {
        if (recipe.length == 0) return true;

        long h1 = 0, h2 = 0;
        for (String token : recipe) {
            Integer tokenValue = tokenId.get(token);
            if (tokenValue == null) return false;
            h1 = (h1 * BASE1 + tokenValue) % MOD1;
            h2 = (h2 * BASE2 + tokenValue) % MOD2;
        }

        Set<Long> hashes = hashesByLen.get(recipe.length);
        return hashes != null && hashes.contains(h1 * MOD2 + h2);
    }
}
```

**Complexity - Part 1:**
| Operation | Time | Space |
|-----------|------|-------|
| Preprocessing | O(n²) | O(n²) |
| Query one recipe | O(m) | O(1) |

---

### Part 2: O(1) Space Solution

**Approach:** Slide through the ingredient list, checking for matches at each position.

**Python Solution:**
```python
def contains_recipe_o1_space(ingredients: list[str], recipe: list[str]) -> bool:
    n, m = len(ingredients), len(recipe)

    if m == 0:
        return True
    if m > n:
        return False

    for start in range(n - m + 1):
        match = True
        for j in range(m):
            if ingredients[start + j] != recipe[j]:
                match = False
                break
        if match:
            return True

    return False


def match_recipes_o1_space(
    ingredients: list[str], recipes: list[list[str]]
) -> list[bool]:
    return [contains_recipe_o1_space(ingredients, recipe) for recipe in recipes]
```

**Java Solution:**
```java
public boolean containsRecipe(String[] ingredients, String[] recipe) {
    int n = ingredients.length;
    int m = recipe.length;

    if (m == 0) return true;
    if (m > n) return false;

    for (int start = 0; start <= n - m; start++) {
        boolean match = true;
        for (int j = 0; j < m; j++) {
            if (!ingredients[start + j].equals(recipe[j])) {
                match = false;
                break;
            }
        }
        if (match) return true;
    }

    return false;
}
```

**Complexity - Part 2:**
| Operation | Time | Space |
|-----------|------|-------|
| Query one recipe | O(n × m) | O(1) |
| Query all recipes | O(n × Σmᵢ) | O(1) |

---

### Part 3: Streaming Solution (Aho-Corasick)

**Approach:** Build a trie from all recipes with failure links. Process stream in O(1) per token.

**Python Solution:**
```python
from collections import deque

class StreamRecipeMatcher:
    def __init__(self, recipes: list[list[str]]):
        self._recipes = recipes
        self._children = [{}]  # node -> {token: next_node}
        self._fail = [0]       # failure links
        self._out = [[]]       # node -> recipe ids ending here
        self._build_trie()
        self._build_fail_links()

    def _new_node(self) -> int:
        self._children.append({})
        self._fail.append(0)
        self._out.append([])
        return len(self._children) - 1

    def _build_trie(self) -> None:
        for recipe_id, recipe in enumerate(self._recipes):
            node = 0
            for token in recipe:
                if token not in self._children[node]:
                    self._children[node][token] = self._new_node()
                node = self._children[node][token]
            self._out[node].append(recipe_id)

    def _build_fail_links(self) -> None:
        q = deque()
        for nxt in self._children[0].values():
            self._fail[nxt] = 0
            q.append(nxt)

        while q:
            node = q.popleft()
            for token, nxt in self._children[node].items():
                f = self._fail[node]
                while f and token not in self._children[f]:
                    f = self._fail[f]
                self._fail[nxt] = self._children[f].get(token, 0)
                self._out[nxt].extend(self._out[self._fail[nxt]])
                q.append(nxt)

    def match_stream(self, ingredient_stream) -> list[bool]:
        matched = [False] * len(self._recipes)
        remaining = len(self._recipes)
        state = 0

        # Handle empty recipes
        for recipe_id in self._out[0]:
            if not matched[recipe_id]:
                matched[recipe_id] = True
                remaining -= 1

        if remaining == 0:
            return matched

        for token in ingredient_stream:
            # Follow fail links until we find a match or reach root
            while state and token not in self._children[state]:
                state = self._fail[state]
            state = self._children[state].get(token, 0)

            # Mark all recipes ending at this state
            for recipe_id in self._out[state]:
                if not matched[recipe_id]:
                    matched[recipe_id] = True
                    remaining -= 1

            if remaining == 0:
                break

        return matched
```

**How Aho-Corasick Works:**
```
Recipes: ["he", "she", "his", "hers"]

Trie Structure:
        root
       / | \
      h  s  (other)
     /|   \
    e i    h
    |      |
    r      e
    |
    s

Fail Links: Connect partial matches to longest proper suffix
- "she" fails to "he" (shares "he" suffix)
- Allows O(1) transitions per character
```

**Complexity - Part 3:**
| Operation | Time | Space |
|-----------|------|-------|
| Build automaton | O(total recipe tokens) | O(total recipe tokens) |
| Process stream | O(stream length) | O(1) per token |

---

### Summary: When to Use Each Approach

| Scenario | Best Approach | Trade-off |
|----------|---------------|-----------|
| Many queries, ingredients fit in memory | Part 1 (Hash) | O(n²) preprocessing, O(m) query |
| Memory constrained, few queries | Part 2 (Two Pointer) | O(n×m) per query, O(1) space |
| Streaming data, can't store ingredients | Part 3 (Aho-Corasick) | O(1) per stream token |

---

## 6. Debugging Service Failures

**Problem Context:**
Investigate a system crash by analyzing logs and service dependencies.

**Interview Structure:**
| Part | Task | Technique |
|------|------|-----------|
| Part 1 | Find first error log | Binary Search |
| Part 2 | Find all affected services | BFS/DFS on reverse graph |
| Part 3 | Find longest failure chain | DFS with memoization |

---

### Part 1: Finding the First Error (Binary Search)

**Problem:**
Given a list of log strings tagged with `[Info]`, `[Warn]`, or `[Error]`:
- Once an `[Error]` appears, all subsequent logs are also `[Error]`
- The line before the first `[Error]` is always `[Warn]`

Find the index of the first `[Error]`. Return `-1` if none exists.

**Example:**
```python
logs = [
    "[Info] boot",
    "[Info] warmup",
    "[Warn] timeout retries high",
    "[Error] downstream unavailable",
    "[Error] service unhealthy",
]
# Output: 3
```

**Key Insight:** The logs are partitioned—all non-errors come before all errors. Use binary search to find the transition point.

**Python Solution:**
```python
def first_error_index(logs: list[str]) -> int:
    def is_error(log_line: str) -> bool:
        return log_line.startswith("[Error]")

    left, right = 0, len(logs) - 1
    answer = -1

    while left <= right:
        mid = (left + right) // 2
        if is_error(logs[mid]):
            answer = mid
            right = mid - 1  # Search left for earlier error
        else:
            left = mid + 1   # Search right for first error

    return answer
```

**Java Solution:**
```java
public int firstErrorIndex(String[] logs) {
    int left = 0, right = logs.length - 1;
    int answer = -1;

    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (logs[mid].startsWith("[Error]")) {
            answer = mid;
            right = mid - 1;
        } else {
            left = mid + 1;
        }
    }

    return answer;
}
```

**Complexity - Part 1:**
| Metric | Complexity |
|--------|------------|
| Time | O(log n) |
| Space | O(1) |

---

### Part 2: Finding All Affected Services (BFS)

**Problem:**
Given a service dependency graph and a failed service, find all services that will fail.

**Rule:** If a service crashes, any service that depends on it (directly or indirectly) also fails.

**Example:**
```python
calls = {
    "A": ["B", "C"],  # A calls B and C
    "B": ["D"],
    "C": ["D"],
    "E": ["A"],
    "F": ["C"],
}
first_error_service = "D"
# Output: {"D", "B", "C", "A", "E", "F"}
```

**Key Insight:** Build a reverse graph (callee → callers) and BFS from the failed service.

```
Original:        Reverse:
E → A → B → D    D → B → A → E
    ↓   ↓            ↓   ↓
F → C ──┘        D → C → A
                     ↓
                     F
```

**Python Solution:**
```python
from collections import defaultdict, deque

def impacted_services(
    calls: dict[str, list[str]],
    first_error_service: str,
) -> set[str]:
    # Build reverse graph: callee -> list of callers
    reverse_graph: dict[str, list[str]] = defaultdict(list)
    for caller, callees in calls.items():
        for callee in callees:
            reverse_graph[callee].append(caller)

    # BFS from failed service
    impacted: set[str] = {first_error_service}
    queue = deque([first_error_service])

    while queue:
        service = queue.popleft()
        for caller in reverse_graph.get(service, []):
            if caller not in impacted:
                impacted.add(caller)
                queue.append(caller)

    return impacted
```

**Java Solution:**
```java
import java.util.*;

public Set<String> impactedServices(
    Map<String, List<String>> calls,
    String firstErrorService
) {
    // Build reverse graph
    Map<String, List<String>> reverseGraph = new HashMap<>();
    for (Map.Entry<String, List<String>> entry : calls.entrySet()) {
        String caller = entry.getKey();
        for (String callee : entry.getValue()) {
            reverseGraph.computeIfAbsent(callee, k -> new ArrayList<>())
                        .add(caller);
        }
    }

    // BFS
    Set<String> impacted = new HashSet<>();
    Queue<String> queue = new LinkedList<>();
    impacted.add(firstErrorService);
    queue.offer(firstErrorService);

    while (!queue.isEmpty()) {
        String service = queue.poll();
        for (String caller : reverseGraph.getOrDefault(service, List.of())) {
            if (!impacted.contains(caller)) {
                impacted.add(caller);
                queue.offer(caller);
            }
        }
    }

    return impacted;
}
```

**Complexity - Part 2:**
| Metric | Complexity |
|--------|------------|
| Time | O(V + E) |
| Space | O(V + E) |

---

### Part 3: Finding the Longest Failure Chain (DFS)

**Problem:**
Find the longest path of cascading failures starting from the failed service.

**Example:**
```
D fails → B fails → A fails → E fails
Output: ["D", "B", "A", "E"]
```

**Key Insight:** Use DFS with memoization on the reverse graph. Since dependencies form a DAG (typically), we can cache results.

**Python Solution:**
```python
from collections import defaultdict

def longest_error_chain(
    calls: dict[str, list[str]],
    first_error_service: str,
) -> list[str]:
    # Build reverse graph
    reverse_graph: dict[str, list[str]] = defaultdict(list)
    for caller, callees in calls.items():
        for callee in callees:
            reverse_graph[callee].append(caller)

    memo: dict[str, list[str]] = {}
    state: dict[str, int] = {}  # 0=unseen, 1=visiting, 2=done

    def dfs(service: str) -> list[str]:
        # Cycle detection
        if state.get(service) == 1:
            return [service]  # or raise error
        if state.get(service) == 2:
            return memo[service]

        state[service] = 1
        best_chain = [service]

        for caller in reverse_graph.get(service, []):
            candidate = [service] + dfs(caller)
            if len(candidate) > len(best_chain):
                best_chain = candidate

        state[service] = 2
        memo[service] = best_chain
        return best_chain

    return dfs(first_error_service)
```

**Java Solution:**
```java
import java.util.*;

public class ServiceDebugger {
    private Map<String, List<String>> reverseGraph;
    private Map<String, List<String>> memo;
    private Map<String, Integer> state;  // 0=unseen, 1=visiting, 2=done

    public List<String> longestErrorChain(
        Map<String, List<String>> calls,
        String firstErrorService
    ) {
        // Build reverse graph
        reverseGraph = new HashMap<>();
        for (Map.Entry<String, List<String>> entry : calls.entrySet()) {
            String caller = entry.getKey();
            for (String callee : entry.getValue()) {
                reverseGraph.computeIfAbsent(callee, k -> new ArrayList<>())
                            .add(caller);
            }
        }

        memo = new HashMap<>();
        state = new HashMap<>();
        return dfs(firstErrorService);
    }

    private List<String> dfs(String service) {
        if (state.getOrDefault(service, 0) == 1) {
            return List.of(service);  // Cycle
        }
        if (state.getOrDefault(service, 0) == 2) {
            return memo.get(service);
        }

        state.put(service, 1);
        List<String> bestChain = new ArrayList<>(List.of(service));

        for (String caller : reverseGraph.getOrDefault(service, List.of())) {
            List<String> candidate = new ArrayList<>();
            candidate.add(service);
            candidate.addAll(dfs(caller));
            if (candidate.size() > bestChain.size()) {
                bestChain = candidate;
            }
        }

        state.put(service, 2);
        memo.put(service, bestChain);
        return bestChain;
    }
}
```

**Walkthrough:**
```
calls = {A: [B,C], B: [D], C: [D], E: [A], F: [C]}
first_error_service = "D"

Reverse graph: {D: [B,C], B: [A], C: [A,F], A: [E]}

DFS from D:
  D → B → A → E (length 4)
  D → C → A → E (length 4)
  D → C → F     (length 3)

Longest chain: [D, B, A, E] or [D, C, A, E]
```

**Complexity - Part 3:**
| Metric | Complexity |
|--------|------------|
| Time | O(V + E) for DAG |
| Space | O(V + E) |

---

### Summary

| Part | Problem | Key Technique | Time |
|------|---------|---------------|------|
| 1 | First error in logs | Binary Search | O(log n) |
| 2 | All affected services | BFS on reverse graph | O(V + E) |
| 3 | Longest failure chain | DFS + memoization | O(V + E) |

**Important Notes:**
- If dependencies have cycles, use Strongly Connected Components (SCC) first
- Part 2 and 3 both require building the reverse dependency graph
- The reverse graph transforms "A calls B" into "B is depended on by A"

---

## 7. Finding the Closest Cake and Optimal Matching

**Problem Context:**
You are given an array representing a line of positions.

**Interview Structure:**
| Part | Array Values | Task |
|------|--------------|------|
| Task 1 | 0=empty, 1=cake | Find distance to nearest cake from a start position |
| Task 2 | 0=empty, 1=person, 2=cake | Pair each person with a unique cake, minimizing total distance |

---

### Task 1: Nearest Cake Distance

**Problem:**
Given a binary array `A` (1=cake, 0=empty) and a `start` index, find the shortest distance to any cake. Return `-1` if no cakes exist.

**Example:**
```python
A = [0, 0, 1, 0, 0, 1, 0]
start = 0
# Output: 2 (closest cake at index 2)
```

**Key Insight:** Expand outward from start in both directions simultaneously.

**Python Solution:**
```python
def nearest_cake_distance(A: list[int], start: int) -> int:
    n = len(A)
    if start < 0 or start >= n:
        raise ValueError("start out of range")

    # Search left
    left = start
    while left >= 0 and A[left] != 1:
        left -= 1

    # Search right
    right = start
    while right < n and A[right] != 1:
        right += 1

    best = float("inf")

    if left >= 0:
        best = min(best, start - left)
    if right < n:
        best = min(best, right - start)

    return -1 if best == float("inf") else int(best)
```

**Java Solution:**
```java
public int nearestCakeDistance(int[] A, int start) {
    int n = A.length;
    if (start < 0 || start >= n) {
        throw new IllegalArgumentException("start out of range");
    }

    // Search left
    int left = start;
    while (left >= 0 && A[left] != 1) {
        left--;
    }

    // Search right
    int right = start;
    while (right < n && A[right] != 1) {
        right++;
    }

    int best = Integer.MAX_VALUE;

    if (left >= 0) {
        best = Math.min(best, start - left);
    }
    if (right < n) {
        best = Math.min(best, right - start);
    }

    return best == Integer.MAX_VALUE ? -1 : best;
}
```

**Complexity - Task 1:**
| Metric | Complexity |
|--------|------------|
| Time | O(n) |
| Space | O(1) |

---

### Task 2: Global Optimal Matching

**Problem:**
Array contains persons (1), cakes (2), and empty spots (0). Pair each person with exactly one unique cake to minimize total distance.

**Why Greedy Fails:**
```python
line = [1, 2, 0, 1, 0, 0, 2]
#       ^  ^     ^        ^
#       P0 C1    P3       C6

# Greedy (each person takes nearest):
#   P0 wants C1 (dist 1)
#   P3 wants C1 (dist 2) ← conflict!

# Optimal global assignment:
#   P0 → C1 (dist 1)
#   P3 → C6 (dist 3)
#   Total: 4
```

**Key Insight:** Use DP. Since persons and cakes are sorted by position, we can match them optimally with `dp[i][j]` = min cost to match first `i` persons using first `j` cakes.

**Recurrence:**
```
dp[i][j] = min(
    dp[i][j-1],                              # skip cake j
    dp[i-1][j-1] + |person[i] - cake[j]|     # assign cake j to person i
)
```

**Python Solution:**
```python
def assign_cakes_globally(line: list[int]) -> dict[int, int]:
    """
    Returns mapping: person_index -> assigned_cake_index
    """
    persons = [i for i, v in enumerate(line) if v == 1]
    cakes = [i for i, v in enumerate(line) if v == 2]

    p, c = len(persons), len(cakes)

    if p == 0:
        return {}
    if p > c:
        raise ValueError("impossible: fewer cakes than persons")

    INF = float("inf")
    # dp[i][j] = min cost to match i persons using first j cakes
    dp = [[INF] * (c + 1) for _ in range(p + 1)]
    # take[i][j] = True if we paired person i with cake j
    take = [[False] * (c + 1) for _ in range(p + 1)]

    # Base case: 0 cost to match 0 persons
    for j in range(c + 1):
        dp[0][j] = 0

    for i in range(1, p + 1):
        for j in range(1, c + 1):
            # Option 1: Skip cake j-1
            best = dp[i][j - 1]
            choose_take = False

            # Option 2: Pair person i-1 with cake j-1
            cost = dp[i - 1][j - 1] + abs(persons[i - 1] - cakes[j - 1])
            if cost < best:
                best = cost
                choose_take = True

            dp[i][j] = best
            take[i][j] = choose_take

    # Backtrack to reconstruct assignment
    assignment: dict[int, int] = {}
    i, j = p, c
    while i > 0 and j > 0:
        if take[i][j]:
            assignment[persons[i - 1]] = cakes[j - 1]
            i -= 1
            j -= 1
        else:
            j -= 1

    return assignment


def assigned_cake_for_person(line: list[int], person_index: int) -> int:
    assignment = assign_cakes_globally(line)
    if person_index not in assignment:
        raise ValueError("person index not found")
    return assignment[person_index]
```

**Java Solution:**
```java
import java.util.*;

public class CakeMatcher {
    public Map<Integer, Integer> assignCakesGlobally(int[] line) {
        List<Integer> persons = new ArrayList<>();
        List<Integer> cakes = new ArrayList<>();

        for (int i = 0; i < line.length; i++) {
            if (line[i] == 1) persons.add(i);
            else if (line[i] == 2) cakes.add(i);
        }

        int p = persons.size(), c = cakes.size();

        if (p == 0) return new HashMap<>();
        if (p > c) throw new IllegalArgumentException("fewer cakes than persons");

        long INF = Long.MAX_VALUE / 2;
        long[][] dp = new long[p + 1][c + 1];
        boolean[][] take = new boolean[p + 1][c + 1];

        for (int i = 0; i <= p; i++) {
            Arrays.fill(dp[i], INF);
        }
        for (int j = 0; j <= c; j++) {
            dp[0][j] = 0;
        }

        for (int i = 1; i <= p; i++) {
            for (int j = 1; j <= c; j++) {
                // Skip cake j-1
                long best = dp[i][j - 1];
                boolean chooseTake = false;

                // Pair person i-1 with cake j-1
                long cost = dp[i - 1][j - 1] +
                    Math.abs(persons.get(i - 1) - cakes.get(j - 1));
                if (cost < best) {
                    best = cost;
                    chooseTake = true;
                }

                dp[i][j] = best;
                take[i][j] = chooseTake;
            }
        }

        // Backtrack
        Map<Integer, Integer> assignment = new HashMap<>();
        int i = p, j = c;
        while (i > 0 && j > 0) {
            if (take[i][j]) {
                assignment.put(persons.get(i - 1), cakes.get(j - 1));
                i--;
                j--;
            } else {
                j--;
            }
        }

        return assignment;
    }
}
```

**Walkthrough:**
```
line = [1, 2, 0, 1, 0, 0, 2]
persons = [0, 3]  (indices where value = 1)
cakes = [1, 6]    (indices where value = 2)

DP Table (dp[i][j] = min cost for i persons, j cakes):
        j=0   j=1   j=2
i=0      0     0     0
i=1     INF    1     1    (person 0 → cake 1, cost=1)
i=2     INF   INF    4    (person 3 → cake 6, cost=3; total=4)

Backtrack from dp[2][2]:
  take[2][2] = True → assign person 3 → cake 6
  take[1][1] = True → assign person 0 → cake 1

Result: {0: 1, 3: 6}
```

**Complexity - Task 2:**
| Metric | Complexity |
|--------|------------|
| Time | O(P × C) |
| Space | O(P × C) |

Where P = number of persons, C = number of cakes.

---

### Optimization Tips

| Scenario | Approach |
|----------|----------|
| Single query for nearest cake | Task 1: Two-pointer O(n) |
| Multiple queries, same array | Precompute nearest cake for all positions in O(n) |
| Global matching, many queries | Run DP once, cache assignment map, answer in O(1) |

**Space Optimization:** If only the total cost is needed (not the assignment), reduce space to O(C) by keeping only two rows of the DP table.

---

## 8. Maximum Number of Events That Can Be Attended

**Problem Statement:**
Given a list of events where each event is `[startDay, endDay]`:
- You can attend an event on any day within its range (inclusive)
- You can only attend **one event per day**

Find the maximum number of events you can attend.

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `[[1,2],[2,3],[3,4]]` | 3 | Day 1→Event 1, Day 2→Event 2, Day 3→Event 3 |
| `[[1,2],[2,3],[3,4],[1,2]]` | 4 | All 4 events can be attended on different days |

**Constraints:**
- `1 <= events.length <= 10^5`
- `1 <= startDay <= endDay <= 10^5`

**Key Insight:**
Use a greedy approach: **always attend the event that ends soonest**. Events ending soon expire quickly—if we skip them, we lose the chance forever. Events ending later can wait.

**Algorithm:**
1. **Sort** events by start day
2. **Min-Heap** tracks end days of currently available events
3. For each day:
   - Add all events starting on this day to the heap
   - Remove expired events (end day < current day)
   - Attend the event ending soonest (pop from heap)

**Python Solution:**
```python
import heapq

def maxEvents(events: list[list[int]]) -> int:
    # Sort by start day
    events.sort(key=lambda x: x[0])

    min_heap = []  # stores end days
    count = 0
    event_idx = 0
    n = len(events)

    day = 1
    while event_idx < n or min_heap:
        # If heap is empty, jump to the next event's start day
        if not min_heap:
            day = events[event_idx][0]

        # Add all events starting on this day
        while event_idx < n and events[event_idx][0] == day:
            heapq.heappush(min_heap, events[event_idx][1])
            event_idx += 1

        # Remove expired events
        while min_heap and min_heap[0] < day:
            heapq.heappop(min_heap)

        # Attend the event ending soonest
        if min_heap:
            heapq.heappop(min_heap)
            count += 1

        day += 1

    return count
```

**Java Solution:**
```java
import java.util.*;

class Solution {
    public int maxEvents(int[][] events) {
        // Sort events by startDay
        Arrays.sort(events, (a, b) -> Integer.compare(a[0], b[0]));

        // Min-Heap to store endDay of available events
        PriorityQueue<Integer> pq = new PriorityQueue<>();

        int count = 0;
        int eventIndex = 0;
        int n = events.length;
        int day = 1;

        while (eventIndex < n || !pq.isEmpty()) {
            // Jump to next event's start if heap is empty
            if (pq.isEmpty()) {
                day = events[eventIndex][0];
            }

            // Add all events starting on this day
            while (eventIndex < n && events[eventIndex][0] == day) {
                pq.offer(events[eventIndex][1]);
                eventIndex++;
            }

            // Remove expired events
            while (!pq.isEmpty() && pq.peek() < day) {
                pq.poll();
            }

            // Attend event ending soonest
            if (!pq.isEmpty()) {
                pq.poll();
                count++;
            }

            day++;
        }

        return count;
    }
}
```

**TypeScript Solution:**
```typescript
function maxEvents(events: number[][]): number {
    // Sort by start day
    events.sort((a, b) => a[0] - b[0]);

    // Simple min-heap implementation using array
    const heap: number[] = [];

    const push = (val: number) => {
        heap.push(val);
        let i = heap.length - 1;
        while (i > 0) {
            const parent = Math.floor((i - 1) / 2);
            if (heap[parent] <= heap[i]) break;
            [heap[parent], heap[i]] = [heap[i], heap[parent]];
            i = parent;
        }
    };

    const pop = (): number => {
        const result = heap[0];
        const last = heap.pop()!;
        if (heap.length > 0) {
            heap[0] = last;
            let i = 0;
            while (true) {
                const left = 2 * i + 1, right = 2 * i + 2;
                let smallest = i;
                if (left < heap.length && heap[left] < heap[smallest]) smallest = left;
                if (right < heap.length && heap[right] < heap[smallest]) smallest = right;
                if (smallest === i) break;
                [heap[i], heap[smallest]] = [heap[smallest], heap[i]];
                i = smallest;
            }
        }
        return result;
    };

    let count = 0;
    let eventIdx = 0;
    let day = 1;

    while (eventIdx < events.length || heap.length > 0) {
        if (heap.length === 0) {
            day = events[eventIdx][0];
        }

        while (eventIdx < events.length && events[eventIdx][0] === day) {
            push(events[eventIdx][1]);
            eventIdx++;
        }

        while (heap.length > 0 && heap[0] < day) {
            pop();
        }

        if (heap.length > 0) {
            pop();
            count++;
        }

        day++;
    }

    return count;
}
```

**Walkthrough - Example 1:**
```
events = [[1,2],[2,3],[3,4]]  (already sorted)

Day 1:
  - Add event [1,2] → heap = [2]
  - Attend event ending day 2 → heap = [], count = 1

Day 2:
  - Add event [2,3] → heap = [3]
  - Attend event ending day 3 → heap = [], count = 2

Day 3:
  - Add event [3,4] → heap = [4]
  - Attend event ending day 4 → heap = [], count = 3

Result: 3
```

**Walkthrough - Example 2:**
```
events = [[1,2],[2,3],[3,4],[1,2]]
sorted = [[1,2],[1,2],[2,3],[3,4]]

Day 1:
  - Add [1,2] and [1,2] → heap = [2, 2]
  - Attend one → heap = [2], count = 1

Day 2:
  - Add [2,3] → heap = [2, 3]
  - Attend event ending day 2 → heap = [3], count = 2

Day 3:
  - Add [3,4] → heap = [3, 4]
  - Attend event ending day 3 → heap = [4], count = 3

Day 4:
  - Attend event ending day 4 → heap = [], count = 4

Result: 4
```

**Why Greedy Works:**
```
Consider events A=[1,3] and B=[2,2]

Wrong approach (attend longer event first):
  Day 1: Attend A
  Day 2: B expires! We can only attend 1 event.

Correct approach (attend event ending soonest):
  Day 1: Skip (B not available yet)
  Day 2: Attend B (ends soonest)
  Day 3: Attend A
  Total: 2 events
```

**Complexity Analysis:**

| Metric | Complexity |
|--------|------------|
| Time | O(n log n) - sorting + heap operations |
| Space | O(n) - heap storage |

**Note:** LeetCode problem #1353

---

## 9. Tracking Top Selling Books

**Problem Statement:**
Build a `BestSellerTracker` class that tracks cumulative book sales and returns top-selling books after each update.

**Class Methods:**
| Method | Description |
|--------|-------------|
| `BestSellerTracker()` | Initialize with no sales data |
| `bestSellers(sales, k)` | Add new sales to totals, return top k books |

**Ranking Rules:**
1. **Higher sales first** (descending)
2. **Tie-breaker:** Alphabetically larger title first ("beta" > "alpha")

If `k` exceeds total books, return all books in order.

**Examples:**

**Case 1:**
```
tracker = BestSellerTracker()

tracker.bestSellers({"a":5, "b":10, "c":15}, 2)
# Totals: {a:5, b:10, c:15}
# Output: ["c", "b"]

tracker.bestSellers({"a":20, "b":20, "c":5}, 2)
# Totals: {a:25, b:30, c:20}
# Output: ["b", "a"]
```

**Case 2:**
```
tracker.bestSellers({"alpha":4, "beta":4, "gamma":1}, 5)
# Totals: {alpha:4, beta:4, gamma:1}
# Tie at 4: "beta" > "alpha" alphabetically
# Output: ["beta", "alpha", "gamma"]
```

**Key Insight:**
Maintain a hash map for cumulative totals. For top-k queries, either sort all books or use a min-heap of size k.

**Python Solution:**
```python
from collections import defaultdict
import heapq

class BestSellerTracker:
    def __init__(self):
        self.totals: dict[str, int] = defaultdict(int)

    def bestSellers(self, sales: dict[str, int], k: int) -> list[str]:
        # Update cumulative totals
        for book, count in sales.items():
            self.totals[book] += count

        # Sort by (-sales, -title) for descending sales, descending alpha
        # Using negative for max behavior
        books = list(self.totals.keys())
        books.sort(key=lambda x: (-self.totals[x], x), reverse=True)

        # reverse=True with key x gives us: highest sales, then "largest" alpha
        # Actually, let's be more explicit:
        books.sort(key=lambda x: (-self.totals[x], [-ord(c) for c in x]))

        return books[:k]


# Cleaner alternative using custom comparator
class BestSellerTrackerV2:
    def __init__(self):
        self.totals: dict[str, int] = defaultdict(int)

    def bestSellers(self, sales: dict[str, int], k: int) -> list[str]:
        for book, count in sales.items():
            self.totals[book] += count

        # Sort: primary = sales desc, secondary = title desc (reverse alpha)
        books = sorted(
            self.totals.keys(),
            key=lambda x: (-self.totals[x], x),
            reverse=False
        )
        # With key=(-sales, title) and reverse=False:
        # -sales sorts descending, title sorts ascending
        # We need title descending, so:
        books = sorted(
            self.totals.keys(),
            key=lambda x: (-self.totals[x], tuple(-ord(c) for c in x))
        )

        return books[:k]


# Simplest correct version
class BestSellerTrackerSimple:
    def __init__(self):
        self.totals: dict[str, int] = defaultdict(int)

    def bestSellers(self, sales: dict[str, int], k: int) -> list[str]:
        for book, count in sales.items():
            self.totals[book] += count

        # Create list of (sales, title) and sort with custom logic
        items = [(self.totals[book], book) for book in self.totals]

        # Sort: higher sales first, then larger title first
        items.sort(key=lambda x: (-x[0], x[1]), reverse=True)
        # reverse=True on tuple: for equal -sales, larger title comes first

        return [item[1] for item in items][:k]
```

**Java Solution:**
```java
import java.util.*;

class BestSellerTracker {
    private Map<String, Integer> totals;

    public BestSellerTracker() {
        totals = new HashMap<>();
    }

    public List<String> bestSellers(Map<String, Integer> sales, int k) {
        // Update cumulative totals
        for (Map.Entry<String, Integer> entry : sales.entrySet()) {
            totals.merge(entry.getKey(), entry.getValue(), Integer::sum);
        }

        // Create list of books and sort
        List<String> books = new ArrayList<>(totals.keySet());

        // Sort: higher sales first, then reverse alphabetical for ties
        books.sort((a, b) -> {
            int salesA = totals.get(a);
            int salesB = totals.get(b);

            if (salesA != salesB) {
                return salesB - salesA;  // Descending by sales
            }
            return b.compareTo(a);  // Descending alphabetical (beta > alpha)
        });

        // Return top k (or all if k > size)
        return books.subList(0, Math.min(k, books.size()));
    }
}
```

**TypeScript Solution:**
```typescript
class BestSellerTracker {
    private totals: Map<string, number>;

    constructor() {
        this.totals = new Map();
    }

    bestSellers(sales: Record<string, number>, k: number): string[] {
        // Update cumulative totals
        for (const [book, count] of Object.entries(sales)) {
            this.totals.set(book, (this.totals.get(book) || 0) + count);
        }

        // Sort books by sales (desc), then by title (desc)
        const books = Array.from(this.totals.keys());

        books.sort((a, b) => {
            const salesA = this.totals.get(a)!;
            const salesB = this.totals.get(b)!;

            if (salesA !== salesB) {
                return salesB - salesA;  // Descending by sales
            }
            return b.localeCompare(a);  // Descending alphabetical
        });

        return books.slice(0, k);
    }
}
```

**Optimized Solution with Min-Heap (for large datasets):**
```python
import heapq
from collections import defaultdict

class BestSellerTrackerOptimized:
    """
    Use min-heap of size k for O(n log k) instead of O(n log n) sorting.
    """
    def __init__(self):
        self.totals: dict[str, int] = defaultdict(int)

    def bestSellers(self, sales: dict[str, int], k: int) -> list[str]:
        for book, count in sales.items():
            self.totals[book] += count

        # Min-heap: keep k largest
        # Heap element: (sales, title) with inverted comparison
        heap = []

        for book, total in self.totals.items():
            # Push (sales, title) - Python compares tuples lexicographically
            # We want min-heap to evict smallest, so use (sales, -title_order)
            # Simpler: just push and maintain size k
            if len(heap) < k:
                heapq.heappush(heap, (total, book))
            elif (total, book) > heap[0]:
                heapq.heapreplace(heap, (total, book))

        # Extract and sort the k elements
        result = []
        while heap:
            result.append(heapq.heappop(heap))

        # Sort by (-sales, reverse_title)
        result.sort(key=lambda x: (-x[0], x[1]), reverse=True)

        return [book for _, book in result]
```

**Walkthrough - Case 1:**
```
Initial: totals = {}

bestSellers({"a":5, "b":10, "c":15}, 2):
  totals = {"a":5, "b":10, "c":15}
  Sorted: [("c",15), ("b",10), ("a",5)]
  Top 2: ["c", "b"]

bestSellers({"a":20, "b":20, "c":5}, 2):
  totals = {"a":25, "b":30, "c":20}
  Sorted: [("b",30), ("a",25), ("c",20)]
  Top 2: ["b", "a"]
```

**Walkthrough - Case 2 (Tie-breaker):**
```
bestSellers({"alpha":4, "beta":4, "gamma":1}, 5):
  totals = {"alpha":4, "beta":4, "gamma":1}

  Sorting with ties:
    "alpha" vs "beta": same sales (4)
    Tie-breaker: "beta" > "alpha" alphabetically
    → "beta" comes first

  Result: ["beta", "alpha", "gamma"]
```

**Complexity Analysis:**

| Approach | Time per Query | Space |
|----------|----------------|-------|
| Full Sort | O(n log n) | O(n) |
| Min-Heap (size k) | O(n log k) | O(n + k) |

Where n = total unique books tracked.

**Follow-up Optimizations:**
- **Frequent queries, rare updates:** Use a balanced BST (TreeMap/TreeSet) for O(log n) updates and O(k) top-k retrieval
- **Real-time leaderboard:** Use a skip list or order-statistic tree

---

## 10. Top K Trending Hashtags

**Problem Statement:**
Given a list of events where each event is `[userId, hashtag]`, find the top k trending hashtags.

**Popularity Rule:** A hashtag's popularity = number of **unique users** who posted it. Duplicate posts by the same user count only once.

**Sorting Rules:**
1. Higher popularity first
2. Tie-breaker: Smaller alphabetically (lexicographically) first

If `k` exceeds total unique hashtags, return all in order.

**Examples:**

**Example 1:**
```
events = [
    ["u1","#ai"], ["u1","#ai"],  // u1 posts #ai twice (counts as 1)
    ["u2","#ai"], ["u2","#ml"],
    ["u3","#ml"],
    ["u4","#db"], ["u4","#db"]
]
k = 2

Popularity:
  #ai: 2 users (u1, u2)
  #ml: 2 users (u2, u3)
  #db: 1 user (u4)

Tie at 2: #ai < #ml alphabetically → #ai first

Output: ["#ai", "#ml"]
```

**Example 2:**
```
events = [
    ["alice","#x"], ["bob","#y"],
    ["alice","#y"], ["alice","#z"]
]
k = 5

Popularity:
  #y: 2 users (alice, bob)
  #x: 1 user (alice)
  #z: 1 user (alice)

Tie at 1: #x < #z alphabetically

Output: ["#y", "#x", "#z"]
```

**Key Insight:**
Use a hash map of sets: `hashtag → set of userIds`. The set size is the popularity score.

**Python Solution:**
```python
from collections import defaultdict

def topKHashtags(events: list[list[str]], k: int) -> list[str]:
    # Map hashtag -> set of unique users
    hashtag_users: dict[str, set[str]] = defaultdict(set)

    for user_id, hashtag in events:
        hashtag_users[hashtag].add(user_id)

    # Calculate popularity (unique user count)
    popularity = {tag: len(users) for tag, users in hashtag_users.items()}

    # Sort: popularity desc, then alphabetically asc
    hashtags = sorted(
        popularity.keys(),
        key=lambda x: (-popularity[x], x)
    )

    return hashtags[:k]
```

**Java Solution:**
```java
import java.util.*;

class Solution {
    public List<String> topKHashtags(String[][] events, int k) {
        // Map hashtag -> set of unique users
        Map<String, Set<String>> hashtagUsers = new HashMap<>();

        for (String[] event : events) {
            String userId = event[0];
            String hashtag = event[1];
            hashtagUsers
                .computeIfAbsent(hashtag, x -> new HashSet<>())
                .add(userId);
        }

        // Create list of hashtags with popularity
        List<String> hashtags = new ArrayList<>(hashtagUsers.keySet());

        // Sort: popularity desc, then alphabetically asc
        hashtags.sort((a, b) -> {
            int popA = hashtagUsers.get(a).size();
            int popB = hashtagUsers.get(b).size();

            if (popA != popB) {
                return popB - popA;  // Descending popularity
            }
            return a.compareTo(b);   // Ascending alphabetical
        });

        return hashtags.subList(0, Math.min(k, hashtags.size()));
    }
}
```

**TypeScript Solution:**
```typescript
function topKHashtags(events: string[][], k: number): string[] {
    // Map hashtag -> set of unique users
    const hashtagUsers: Map<string, Set<string>> = new Map();

    for (const [userId, hashtag] of events) {
        if (!hashtagUsers.has(hashtag)) {
            hashtagUsers.set(hashtag, new Set());
        }
        hashtagUsers.get(hashtag)!.add(userId);
    }

    // Get hashtags and sort
    const hashtags = Array.from(hashtagUsers.keys());

    hashtags.sort((a, b) => {
        const popA = hashtagUsers.get(a)!.size;
        const popB = hashtagUsers.get(b)!.size;

        if (popA !== popB) {
            return popB - popA;  // Descending popularity
        }
        return a.localeCompare(b);  // Ascending alphabetical
    });

    return hashtags.slice(0, k);
}
```

**Optimized Solution with Min-Heap:**
```python
import heapq
from collections import defaultdict

def topKHashtagsOptimized(events: list[list[str]], k: int) -> list[str]:
    """
    Use min-heap for O(n log k) when k << number of hashtags.
    """
    hashtag_users: dict[str, set[str]] = defaultdict(set)

    for user_id, hashtag in events:
        hashtag_users[hashtag].add(user_id)

    # Min-heap of size k
    # Element: (popularity, hashtag)
    # For ties, we want smaller hashtag to win, so use (pop, hashtag)
    # Min-heap evicts smallest, so we need to invert for max behavior
    heap = []

    for hashtag, users in hashtag_users.items():
        popularity = len(users)

        if len(heap) < k:
            # Push (-popularity, hashtag) for max-heap behavior
            # Actually, let's use (popularity, hashtag) and reverse at end
            heapq.heappush(heap, (popularity, hashtag))
        else:
            # Compare with smallest in heap
            if (popularity, hashtag) > (heap[0][0], heap[0][1]):
                # Wait, we need custom comparison for tie-breaker
                # For tie: smaller hashtag should rank higher (stay in heap)
                pass

            # Simpler: just collect all, then use heapq.nlargest with key
            pass

    # Actually, cleaner approach for this problem:
    items = [(len(users), tag) for tag, users in hashtag_users.items()]

    # nlargest with custom key: (-popularity, hashtag) for sorting
    # But nlargest uses max comparison...

    # Simplest correct approach: sort all
    items.sort(key=lambda x: (-x[0], x[1]))
    return [tag for _, tag in items[:k]]
```

**Cleaner Heap Solution:**
```python
import heapq
from collections import defaultdict

def topKHashtagsHeap(events: list[list[str]], k: int) -> list[str]:
    hashtag_users: dict[str, set[str]] = defaultdict(set)

    for user_id, hashtag in events:
        hashtag_users[hashtag].add(user_id)

    # Use nlargest with custom key
    # Key: (popularity, reversed_hashtag) for proper ordering
    # Or just sort since nlargest with complex keys can be tricky

    hashtags = list(hashtag_users.keys())

    # heapq.nsmallest with inverted key for "nlargest" behavior
    result = heapq.nsmallest(
        k,
        hashtags,
        key=lambda x: (-len(hashtag_users[x]), x)
    )

    return result
```

**Walkthrough - Example 1:**
```
events = [["u1","#ai"],["u1","#ai"],["u2","#ai"],
          ["u2","#ml"],["u3","#ml"],["u4","#db"],["u4","#db"]]

Step 1: Build user sets
  #ai → {u1, u2}
  #ml → {u2, u3}
  #db → {u4}

Step 2: Calculate popularity
  #ai: 2
  #ml: 2
  #db: 1

Step 3: Sort by (-popularity, hashtag)
  (-2, "#ai") < (-2, "#ml") < (-1, "#db")
  → ["#ai", "#ml", "#db"]

Step 4: Take top k=2
  → ["#ai", "#ml"]
```

**Comparison with Previous Problem (#9):**

| Aspect | #9 Best Sellers | #10 Trending Hashtags |
|--------|-----------------|----------------------|
| Count metric | Total sales (cumulative) | Unique users (set size) |
| Tie-breaker | Larger alphabetically | Smaller alphabetically |
| State | Persistent across calls | Single query |

**Complexity Analysis:**

| Approach | Time | Space |
|----------|------|-------|
| Full Sort | O(e + h log h) | O(e) |
| Heap (nsmallest) | O(e + h log k) | O(e) |

Where e = number of events, h = number of unique hashtags.

---

*Last updated: 2026-04-19*
