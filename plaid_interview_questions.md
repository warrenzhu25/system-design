# Plaid Interview Questions

---

## 1. Bank Mapping Lookup

**Problem Statement:**
You are given two mapping datasets:

- **Mapping A (one-to-one)**: each `x` maps to exactly one `y`.
- **Mapping B (one-to-many)**: each `y` maps to one or more bank values (a list/set of banks).

Your task is to connect the two mappings and support fast lookups:

- **Input**: an `x`
- **Output**: the bank (or list of banks) associated with `x`

**Requirements:**
- You may preprocess both mappings once; subsequent queries should be as fast as possible.
- Specify what to return when `x` is missing, or when `y` has no associated banks.
- Implement:
  - `build_index(mapping_a, mapping_b)` returning an index for fast queries
  - `query(index, x)` returning the bank(s) for `x`

**Expected Scale (typical interview assumptions):**
- `len(mapping_a)` and `len(mapping_b)` can be up to 1e5
- Aim for ~O(n) preprocessing and amortized O(1) query time

**Example:**

Mapping A:
```
a1 -> y1
a2 -> y2
```

Mapping B:
```
y1 -> [BofA, Chase]
y2 -> [WellsFargo]
```

Queries:
```
a1 => [BofA, Chase]
a2 => [WellsFargo]
```

**Test Cases (stdin -> stdout):**

| Input | Output |
|-------|--------|
| `A: [(a1,y1),(a2,y2)] B: [(y1,BofA),(y1,Chase),(y2,WellsFargo)] Q: [a1,a2]` | `[[BofA,Chase],[WellsFargo]]` |
| `A: [(a1,y1)] B: [(y2,BofA)] Q: [a1]` | `[[]]` |
| `A: [(a1,y1)] B: [(y1,BofA),(y1,BofA)] Q: [a1]` | `[[BofA]]` |
| `A: [(a1,y1),(a2,y1)] B: [(y1,Chase)] Q: [a1,a2]` | `[[Chase],[Chase]]` |
| `A: [] B: [] Q: [a1]` | `[[]]` |

**Key Insights:**
1. Build a hash map from `y -> set(banks)` from Mapping B (handles duplicates)
2. Build the final index by iterating Mapping A and looking up each `y` in the Mapping B hash map
3. Return empty list for missing `x` or when `y` has no associated banks

**Python Solution:**
```python
def build_index(mapping_a: list[tuple[str, str]],
                mapping_b: list[tuple[str, str]]) -> dict[str, list[str]]:
    """
    Build an index for fast x -> banks lookup.

    Time: O(n + m) where n = len(mapping_a), m = len(mapping_b)
    Space: O(n + m) for storing both mappings
    """
    # Step 1: Build y -> set(banks) from mapping_b
    # Using set to handle duplicate bank entries
    y_to_banks: dict[str, set[str]] = {}
    for y, bank in mapping_b:
        if y not in y_to_banks:
            y_to_banks[y] = set()
        y_to_banks[y].add(bank)

    # Step 2: Build x -> banks by joining through y
    index: dict[str, list[str]] = {}
    for x, y in mapping_a:
        # Get banks for this y, or empty set if y not found
        banks = y_to_banks.get(y, set())
        index[x] = list(banks)

    return index


def query(index: dict[str, list[str]], x: str) -> list[str]:
    """
    Query the index for banks associated with x.

    Time: O(1) amortized
    Space: O(1)

    Returns empty list if x is not found.
    """
    return index.get(x, [])


# Example usage
if __name__ == "__main__":
    # Example from problem
    mapping_a = [("a1", "y1"), ("a2", "y2")]
    mapping_b = [("y1", "BofA"), ("y1", "Chase"), ("y2", "WellsFargo")]

    index = build_index(mapping_a, mapping_b)

    print(query(index, "a1"))  # ['BofA', 'Chase'] (order may vary)
    print(query(index, "a2"))  # ['WellsFargo']
    print(query(index, "a3"))  # [] (not found)
```

**Java Solution:**
```java
import java.util.*;

public class BankMappingLookup {

    /**
     * Build an index for fast x -> banks lookup.
     * Time: O(n + m), Space: O(n + m)
     */
    public static Map<String, List<String>> buildIndex(
            List<String[]> mappingA,
            List<String[]> mappingB) {

        // Step 1: Build y -> set(banks) from mappingB
        Map<String, Set<String>> yToBanks = new HashMap<>();
        for (String[] pair : mappingB) {
            String y = pair[0];
            String bank = pair[1];
            yToBanks.computeIfAbsent(y, k -> new HashSet<>()).add(bank);
        }

        // Step 2: Build x -> banks by joining through y
        Map<String, List<String>> index = new HashMap<>();
        for (String[] pair : mappingA) {
            String x = pair[0];
            String y = pair[1];
            Set<String> banks = yToBanks.getOrDefault(y, Collections.emptySet());
            index.put(x, new ArrayList<>(banks));
        }

        return index;
    }

    /**
     * Query the index for banks associated with x.
     * Time: O(1) amortized
     */
    public static List<String> query(Map<String, List<String>> index, String x) {
        return index.getOrDefault(x, Collections.emptyList());
    }

    public static void main(String[] args) {
        List<String[]> mappingA = Arrays.asList(
            new String[]{"a1", "y1"},
            new String[]{"a2", "y2"}
        );
        List<String[]> mappingB = Arrays.asList(
            new String[]{"y1", "BofA"},
            new String[]{"y1", "Chase"},
            new String[]{"y2", "WellsFargo"}
        );

        Map<String, List<String>> index = buildIndex(mappingA, mappingB);

        System.out.println(query(index, "a1")); // [BofA, Chase]
        System.out.println(query(index, "a2")); // [WellsFargo]
        System.out.println(query(index, "a3")); // []
    }
}
```

**TypeScript Solution:**
```typescript
type Index = Map<string, string[]>;

/**
 * Build an index for fast x -> banks lookup.
 * Time: O(n + m), Space: O(n + m)
 */
function buildIndex(
    mappingA: [string, string][],
    mappingB: [string, string][]
): Index {
    // Step 1: Build y -> set(banks) from mappingB
    const yToBanks = new Map<string, Set<string>>();
    for (const [y, bank] of mappingB) {
        if (!yToBanks.has(y)) {
            yToBanks.set(y, new Set());
        }
        yToBanks.get(y)!.add(bank);
    }

    // Step 2: Build x -> banks by joining through y
    const index: Index = new Map();
    for (const [x, y] of mappingA) {
        const banks = yToBanks.get(y) ?? new Set();
        index.set(x, Array.from(banks));
    }

    return index;
}

/**
 * Query the index for banks associated with x.
 * Time: O(1) amortized
 */
function query(index: Index, x: string): string[] {
    return index.get(x) ?? [];
}

// Example usage
const mappingA: [string, string][] = [["a1", "y1"], ["a2", "y2"]];
const mappingB: [string, string][] = [["y1", "BofA"], ["y1", "Chase"], ["y2", "WellsFargo"]];

const index = buildIndex(mappingA, mappingB);

console.log(query(index, "a1")); // ['BofA', 'Chase']
console.log(query(index, "a2")); // ['WellsFargo']
console.log(query(index, "a3")); // []
```

**Complexity Analysis:**
- **Preprocessing (`build_index`):**
  - Time: O(n + m) where n = len(mapping_a), m = len(mapping_b)
  - Space: O(n + m) to store intermediate y_to_banks and final index
- **Query:**
  - Time: O(1) amortized (hash map lookup)
  - Space: O(1)

**Edge Cases Handled:**
1. `x` not found in mapping_a → return `[]`
2. `y` has no associated banks in mapping_b → return `[]`
3. Duplicate banks for same `y` → deduplicated using set
4. Multiple `x` values mapping to same `y` → each gets same bank list
5. Empty mappings → queries return `[]`

**Follow-up Questions:**
1. **Memory optimization**: If many `x` values map to the same `y`, we could share the same list reference instead of copying.
2. **Streaming updates**: How would you handle incremental updates to either mapping?
3. **Distributed systems**: How would you scale this across multiple machines?

---

## 2. Shopping Cart Coupon Calculator

**Problem Statement:**
You are building a coupon system for an e-commerce platform. Given a shopping cart and a list of coupons, determine which discounts apply to each item.

**Coupon Structure:**
```python
{
    'categories': ['fruit', 'toy'],      # Categories this coupon applies to
    'percent_discount': 15,               # Percentage off (or None)
    'amount_discount': None,              # Fixed amount off (or None)
    'minimum_num_items_required': 2,      # Min items in category (or None)
    'minimum_amount_required': 10.00      # Min spend in category (or None)
}
```

**Validation Rules:**
- Exactly one of `percent_discount` or `amount_discount` must be non-null
- Throw an error if neither is set, or if both are set
- The `minimum_*` fields are optional (can be null)

**Cart Item Structure:**
```python
{'price': 2.00, 'category': 'fruit'}
```

**Requirements:**
1. For each item, find all applicable coupons
2. A coupon applies if:
   - The item's category is in the coupon's `categories` list
   - If `minimum_num_items_required` is set: cart has >= that many items in matching categories
   - If `minimum_amount_required` is set: total spend in matching categories >= that amount
3. Return all applicable discounts per item/category

**Example:**

Cart:
```python
[
    {'price': 2.00, 'category': 'fruit'},
    {'price': 20.00, 'category': 'toy'},
    {'price': 5.00, 'category': 'clothing'},
    {'price': 8.00, 'category': 'fruit'}
]
```

Coupons:
```python
[
    {
        'categories': ['clothing', 'toy'],
        'percent_discount': None,
        'amount_discount': 6,
        'minimum_num_items_required': None,
        'minimum_amount_required': None
    },
    {
        'categories': ['fruit', 'toy'],
        'percent_discount': 15,
        'amount_discount': None,
        'minimum_num_items_required': 2,
        'minimum_amount_required': 10.00
    }
]
```

Analysis:
- **fruit** items: 2 items, total $10.00 → meets coupon 2's requirements (≥2 items, ≥$10)
- **toy** item: qualifies for coupon 1 ($6 off) and coupon 2 (15% off, since fruit+toy combined meets minimums)
- **clothing** item: qualifies for coupon 1 ($6 off)

Output for toy: `[-15%, -$6]` (both coupons apply)

**Key Insights:**
1. Pre-compute category statistics (count, total) for efficient minimum checks
2. Validate coupons upfront (exactly one discount type)
3. For each coupon, check if cart meets minimum requirements across ALL its categories combined
4. Match items to applicable coupons based on category membership

**Python Solution:**
```python
from dataclasses import dataclass
from typing import Optional


@dataclass
class Coupon:
    categories: list[str]
    percent_discount: Optional[float]
    amount_discount: Optional[float]
    minimum_num_items_required: Optional[int]
    minimum_amount_required: Optional[float]

    def __post_init__(self):
        has_percent = self.percent_discount is not None
        has_amount = self.amount_discount is not None
        if has_percent == has_amount:  # Both true or both false
            raise ValueError(
                "Exactly one of percent_discount or amount_discount must be set"
            )


@dataclass
class CartItem:
    price: float
    category: str


@dataclass
class Discount:
    percent_off: Optional[float] = None
    amount_off: Optional[float] = None

    def __repr__(self):
        if self.percent_off is not None:
            return f"-{self.percent_off}%"
        return f"-${self.amount_off}"


def compute_category_stats(
    cart: list[CartItem],
    categories: set[str]
) -> tuple[int, float]:
    """Compute item count and total for given categories."""
    count = 0
    total = 0.0
    for item in cart:
        if item.category in categories:
            count += 1
            total += item.price
    return count, total


def coupon_applies_to_cart(
    coupon: Coupon,
    cart: list[CartItem]
) -> bool:
    """Check if cart meets coupon's minimum requirements."""
    category_set = set(coupon.categories)
    count, total = compute_category_stats(cart, category_set)

    # Check minimum items requirement
    if coupon.minimum_num_items_required is not None:
        if count < coupon.minimum_num_items_required:
            return False

    # Check minimum amount requirement
    if coupon.minimum_amount_required is not None:
        if total < coupon.minimum_amount_required:
            return False

    return True


def get_discounts_for_item(
    item: CartItem,
    coupons: list[Coupon],
    cart: list[CartItem]
) -> list[Discount]:
    """Get all applicable discounts for a single item."""
    discounts = []

    for coupon in coupons:
        # Check if item's category matches coupon
        if item.category not in coupon.categories:
            continue

        # Check if cart meets coupon requirements
        if not coupon_applies_to_cart(coupon, cart):
            continue

        # Coupon applies - create discount
        if coupon.percent_discount is not None:
            discounts.append(Discount(percent_off=coupon.percent_discount))
        else:
            discounts.append(Discount(amount_off=coupon.amount_discount))

    return discounts


def apply_coupons(
    cart: list[CartItem],
    coupons: list[Coupon]
) -> dict[str, list[Discount]]:
    """
    Apply all coupons to cart and return discounts per category.

    Time: O(c * n) where c = number of coupons, n = cart size
    Space: O(c) for storing applicable discounts
    """
    # Group discounts by category
    category_discounts: dict[str, list[Discount]] = {}

    for item in cart:
        discounts = get_discounts_for_item(item, coupons, cart)
        if item.category not in category_discounts:
            category_discounts[item.category] = []
        # Merge discounts (avoid duplicates for same category)
        for d in discounts:
            if d not in category_discounts[item.category]:
                category_discounts[item.category].append(d)

    return category_discounts


# Example usage
if __name__ == "__main__":
    cart = [
        CartItem(price=2.00, category='fruit'),
        CartItem(price=20.00, category='toy'),
        CartItem(price=5.00, category='clothing'),
        CartItem(price=8.00, category='fruit'),
    ]

    coupons = [
        Coupon(
            categories=['clothing', 'toy'],
            percent_discount=None,
            amount_discount=6,
            minimum_num_items_required=None,
            minimum_amount_required=None
        ),
        Coupon(
            categories=['fruit', 'toy'],
            percent_discount=15,
            amount_discount=None,
            minimum_num_items_required=2,
            minimum_amount_required=10.00
        ),
    ]

    result = apply_coupons(cart, coupons)

    for category, discounts in result.items():
        print(f"{category}: {discounts}")

    # Output:
    # fruit: [-15%]
    # toy: [-$6, -15%]
    # clothing: [-$6]
```

**Test Cases:**

| Scenario | Expected |
|----------|----------|
| Item matches coupon category, no minimums | Discount applies |
| Item matches, min_items not met | No discount |
| Item matches, min_amount not met | No discount |
| Item matches, both minimums met | Discount applies |
| Multiple coupons apply to same item | All discounts returned |
| Coupon with both % and $ discount | Throws error |
| Coupon with neither discount | Throws error |

**Complexity Analysis:**
- **Time**: O(c × n) where c = number of coupons, n = cart size
  - For each coupon, we scan the cart to compute category stats
- **Space**: O(c + k) where k = number of unique categories

**Edge Cases Handled:**
1. Invalid coupon (both/neither discount type) → throws error
2. Empty cart → no discounts
3. No matching categories → no discounts
4. Multiple coupons for same category → all returned
5. Minimum requirements span multiple categories in coupon

**Follow-up Questions:**
1. **Optimization**: How would you avoid recomputing category stats for each coupon?
2. **Stacking rules**: What if only one coupon can apply per item?
3. **Best discount**: How would you find the optimal combination of coupons?
4. **Real-time updates**: How to handle cart changes efficiently?

---

## 3. Logger Rate Limiter

**Problem Statement:**
Design a logger system that receives a stream of messages along with their timestamps. Each unique message should only be printed at most every 10 seconds (i.e., a message printed at timestamp `t` will prevent the same message from being printed until timestamp `t + 10`).

Implement the `Logger` class:
- `Logger()` Initializes the logger object
- `bool shouldPrintMessage(int timestamp, string message)` Returns `true` if the message should be printed at the given timestamp, otherwise returns `false`

**Example:**
```
Input:
["Logger", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage", "shouldPrintMessage"]
[[], [1, "foo"], [2, "bar"], [3, "foo"], [8, "bar"], [10, "foo"], [11, "foo"]]

Output:
[null, true, true, false, false, false, true]

Explanation:
Logger logger = new Logger();
logger.shouldPrintMessage(1, "foo");  // return true, next allowed at t=11
logger.shouldPrintMessage(2, "bar");  // return true, next allowed at t=12
logger.shouldPrintMessage(3, "foo");  // return false, 3 < 11
logger.shouldPrintMessage(8, "bar");  // return false, 8 < 12
logger.shouldPrintMessage(10, "foo"); // return false, 10 < 11
logger.shouldPrintMessage(11, "foo"); // return true, 11 >= 11, next allowed at t=21
```

**Key Insights:**
1. Use a hash map to store the next allowed timestamp for each message
2. When a message arrives, check if current time >= next allowed time
3. If allowed, update the next allowed time to `current + 10`

**Python Solution:**
```python
class Logger:
    """
    Rate-limiting logger that allows each message at most once per 10 seconds.

    Time: O(1) per call
    Space: O(m) where m = number of unique messages
    """

    def __init__(self):
        self.message_timestamps: dict[str, int] = {}

    def should_print_message(self, timestamp: int, message: str) -> bool:
        # Check if message can be printed
        if message not in self.message_timestamps:
            # First time seeing this message
            self.message_timestamps[message] = timestamp + 10
            return True

        if timestamp >= self.message_timestamps[message]:
            # Enough time has passed
            self.message_timestamps[message] = timestamp + 10
            return True

        # Rate limited
        return False


# Example usage
if __name__ == "__main__":
    logger = Logger()
    print(logger.should_print_message(1, "foo"))   # True
    print(logger.should_print_message(2, "bar"))   # True
    print(logger.should_print_message(3, "foo"))   # False
    print(logger.should_print_message(8, "bar"))   # False
    print(logger.should_print_message(10, "foo"))  # False
    print(logger.should_print_message(11, "foo"))  # True
```

**Java Solution:**
```java
import java.util.HashMap;
import java.util.Map;

public class Logger {
    private Map<String, Integer> messageTimestamps;

    public Logger() {
        messageTimestamps = new HashMap<>();
    }

    public boolean shouldPrintMessage(int timestamp, String message) {
        if (!messageTimestamps.containsKey(message)) {
            messageTimestamps.put(message, timestamp + 10);
            return true;
        }

        if (timestamp >= messageTimestamps.get(message)) {
            messageTimestamps.put(message, timestamp + 10);
            return true;
        }

        return false;
    }
}
```

**Complexity Analysis:**
- **Time**: O(1) per call (hash map operations)
- **Space**: O(m) where m = number of unique messages seen

**Follow-up Questions:**
1. **Memory cleanup**: How would you handle memory if messages keep accumulating? (Use LRU cache or periodic cleanup)
2. **Distributed systems**: How would you implement this across multiple servers? (Centralized store like Redis)
3. **Variable rate limits**: What if different messages have different rate limits?

---

## 4. Transaction Sync with Cursor-Based Pagination

**Problem Statement:**
You are consuming transactions from a paginated API. Each API response contains:
- A list of transactions (may contain duplicates across pages)
- A `next_cursor` for fetching the next page
- A `has_more` boolean indicating if more pages exist

Implement a function to merge all pages into a single deduplicated list sorted by timestamp.

**Transaction Structure:**
```python
{
    'transaction_id': 'txn_123',
    'timestamp': 1699900000,
    'amount': 50.00,
    'description': 'Coffee Shop'
}
```

**API Response Structure:**
```python
{
    'transactions': [...],
    'next_cursor': 'cursor_abc123',
    'has_more': True
}
```

**Requirements:**
1. Fetch all pages starting from an initial cursor
2. Deduplicate transactions by `transaction_id`
3. Sort final list by `timestamp` (ascending)
4. Handle pagination errors gracefully
5. Return the final cursor for future syncs

**Example:**
```python
# Page 1: cursor="" -> transactions [A, B, C], next_cursor="cur1", has_more=True
# Page 2: cursor="cur1" -> transactions [C, D, E], next_cursor="cur2", has_more=False
# Result: [A, B, C, D, E] sorted by timestamp, final_cursor="cur2"
```

**Python Solution:**
```python
from dataclasses import dataclass
from typing import Callable


@dataclass
class Transaction:
    transaction_id: str
    timestamp: int
    amount: float
    description: str


@dataclass
class SyncResponse:
    transactions: list[Transaction]
    next_cursor: str
    has_more: bool


@dataclass
class SyncResult:
    transactions: list[Transaction]
    cursor: str


def sync_all_transactions(
    fetch_page: Callable[[str], SyncResponse],
    initial_cursor: str = ""
) -> SyncResult:
    """
    Fetch all transaction pages and merge into deduplicated sorted list.

    Args:
        fetch_page: Function that takes cursor and returns SyncResponse
        initial_cursor: Starting cursor (empty string for initial sync)

    Returns:
        SyncResult with deduplicated transactions and final cursor

    Time: O(n log n) where n = total transactions across all pages
    Space: O(n) for storing all transactions
    """
    seen_ids: set[str] = set()
    all_transactions: list[Transaction] = []
    cursor = initial_cursor

    while True:
        response = fetch_page(cursor)

        # Deduplicate as we collect
        for txn in response.transactions:
            if txn.transaction_id not in seen_ids:
                seen_ids.add(txn.transaction_id)
                all_transactions.append(txn)

        cursor = response.next_cursor

        if not response.has_more:
            break

    # Sort by timestamp
    all_transactions.sort(key=lambda t: t.timestamp)

    return SyncResult(transactions=all_transactions, cursor=cursor)


# Simulated API for testing
def create_mock_api() -> Callable[[str], SyncResponse]:
    """Create a mock paginated API."""
    pages = {
        "": SyncResponse(
            transactions=[
                Transaction("txn_1", 1000, 25.00, "Coffee"),
                Transaction("txn_2", 1005, 100.00, "Groceries"),
            ],
            next_cursor="cursor_1",
            has_more=True
        ),
        "cursor_1": SyncResponse(
            transactions=[
                Transaction("txn_2", 1005, 100.00, "Groceries"),  # Duplicate
                Transaction("txn_3", 1002, 50.00, "Gas"),
            ],
            next_cursor="cursor_2",
            has_more=False
        ),
    }

    def fetch(cursor: str) -> SyncResponse:
        return pages.get(cursor, SyncResponse([], "", False))

    return fetch


if __name__ == "__main__":
    mock_api = create_mock_api()
    result = sync_all_transactions(mock_api, "")

    print(f"Final cursor: {result.cursor}")
    print(f"Transactions ({len(result.transactions)}):")
    for txn in result.transactions:
        print(f"  {txn.transaction_id}: {txn.description} @ {txn.timestamp}")

    # Output:
    # Final cursor: cursor_2
    # Transactions (3):
    #   txn_1: Coffee @ 1000
    #   txn_3: Gas @ 1002
    #   txn_2: Groceries @ 1005
```

**Complexity Analysis:**
- **Time**: O(n log n) for sorting, where n = total transactions
- **Space**: O(n) for storing transactions and seen IDs

**Edge Cases:**
1. Empty initial response → return empty list with initial cursor
2. Duplicate transactions across pages → deduplicate by ID
3. API failure mid-pagination → caller should retry from original cursor
4. Single page (has_more=false immediately) → handle correctly

**Follow-up Questions:**
1. **Incremental updates**: How would you handle transaction modifications and deletions?
2. **Memory efficiency**: What if transactions don't fit in memory?
3. **Retry logic**: How would you implement exponential backoff for failures?
4. **Streaming**: How would you process transactions as they arrive instead of collecting all first?

---

## 5. Stock Price Maximum Profit

**Problem Statement:**
You are given an array `prices` where `prices[i]` is the price of a stock on day `i`. Find the maximum profit you can achieve by buying on one day and selling on a later day. You must buy before you sell.

Return `0` if no profit is possible.

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `[7, 1, 5, 3, 6, 4]` | `5` | Buy at 1 (day 2), sell at 6 (day 5) |
| `[7, 6, 4, 3, 1]` | `0` | Prices only decrease, no profit possible |
| `[2, 4, 1]` | `2` | Buy at 2 (day 1), sell at 4 (day 2) |

**Key Insights:**
1. Track the minimum price seen so far
2. At each step, calculate potential profit if selling today
3. Update maximum profit if current profit is higher
4. Single pass through the array

**Python Solution:**
```python
def max_profit(prices: list[int]) -> int:
    """
    Find maximum profit from single buy-sell transaction.

    Time: O(n)
    Space: O(1)
    """
    if not prices:
        return 0

    min_price = float('inf')
    max_profit = 0

    for price in prices:
        if price < min_price:
            min_price = price
        else:
            profit = price - min_price
            max_profit = max(max_profit, profit)

    return max_profit


# Test cases
print(max_profit([7, 1, 5, 3, 6, 4]))  # 5
print(max_profit([7, 6, 4, 3, 1]))     # 0
print(max_profit([2, 4, 1]))           # 2
print(max_profit([]))                   # 0
print(max_profit([5]))                  # 0
```

**Java Solution:**
```java
public int maxProfit(int[] prices) {
    if (prices == null || prices.length == 0) {
        return 0;
    }

    int minPrice = Integer.MAX_VALUE;
    int maxProfit = 0;

    for (int price : prices) {
        if (price < minPrice) {
            minPrice = price;
        } else {
            maxProfit = Math.max(maxProfit, price - minPrice);
        }
    }

    return maxProfit;
}
```

**TypeScript Solution:**
```typescript
function maxProfit(prices: number[]): number {
    if (prices.length === 0) return 0;

    let minPrice = Infinity;
    let maxProfit = 0;

    for (const price of prices) {
        if (price < minPrice) {
            minPrice = price;
        } else {
            maxProfit = Math.max(maxProfit, price - minPrice);
        }
    }

    return maxProfit;
}
```

**Complexity Analysis:**
- **Time**: O(n) single pass
- **Space**: O(1) constant extra space

**Follow-up Questions:**
1. **Multiple transactions**: What if you can buy and sell multiple times? (Greedy: sum all upward movements)
2. **With cooldown**: What if you must wait one day after selling before buying again? (DP)
3. **With transaction fee**: What if each transaction has a fee? (DP with fee consideration)
4. **At most K transactions**: What if you can make at most K transactions? (2D DP)

---

## 6. Valid Parentheses

**Problem Statement:**
Given a string `s` containing just the characters `'('`, `')'`, `'{'`, `'}'`, `'['` and `']'`, determine if the input string is valid.

A string is valid if:
1. Open brackets must be closed by the same type of brackets
2. Open brackets must be closed in the correct order
3. Every close bracket has a corresponding open bracket of the same type

**Examples:**

| Input | Output | Explanation |
|-------|--------|-------------|
| `"()"` | `true` | Simple matching pair |
| `"()[]{}"` | `true` | Multiple matching pairs |
| `"(]"` | `false` | Mismatched types |
| `"([)]"` | `false` | Wrong order |
| `"{[]}"` | `true` | Nested correctly |

**Key Insights:**
1. Use a stack to track open brackets
2. When encountering a closing bracket, check if it matches the top of stack
3. At the end, stack should be empty

**Python Solution:**
```python
def is_valid(s: str) -> bool:
    """
    Check if parentheses string is valid.

    Time: O(n)
    Space: O(n) worst case for stack
    """
    stack = []
    mapping = {')': '(', ']': '[', '}': '{'}

    for char in s:
        if char in mapping:
            # Closing bracket - check match
            if not stack or stack[-1] != mapping[char]:
                return False
            stack.pop()
        else:
            # Opening bracket - push to stack
            stack.append(char)

    return len(stack) == 0


# Test cases
print(is_valid("()"))       # True
print(is_valid("()[]{}"))   # True
print(is_valid("(]"))       # False
print(is_valid("([)]"))     # False
print(is_valid("{[]}"))     # True
print(is_valid(""))         # True
print(is_valid("(("))       # False
```

**Java Solution:**
```java
import java.util.Stack;
import java.util.Map;

public boolean isValid(String s) {
    Stack<Character> stack = new Stack<>();
    Map<Character, Character> mapping = Map.of(
        ')', '(',
        ']', '[',
        '}', '{'
    );

    for (char c : s.toCharArray()) {
        if (mapping.containsKey(c)) {
            if (stack.isEmpty() || stack.peek() != mapping.get(c)) {
                return false;
            }
            stack.pop();
        } else {
            stack.push(c);
        }
    }

    return stack.isEmpty();
}
```

**TypeScript Solution:**
```typescript
function isValid(s: string): boolean {
    const stack: string[] = [];
    const mapping: Record<string, string> = {
        ')': '(',
        ']': '[',
        '}': '{'
    };

    for (const char of s) {
        if (char in mapping) {
            if (stack.length === 0 || stack[stack.length - 1] !== mapping[char]) {
                return false;
            }
            stack.pop();
        } else {
            stack.push(char);
        }
    }

    return stack.length === 0;
}
```

**Complexity Analysis:**
- **Time**: O(n) single pass through string
- **Space**: O(n) worst case when all opening brackets

**Edge Cases:**
1. Empty string → valid
2. Only opening brackets → invalid
3. Only closing brackets → invalid
4. Single character → invalid
5. Deeply nested → valid if properly matched

**Follow-up Questions:**
1. **Minimum additions**: How many brackets must be added to make it valid?
2. **Longest valid substring**: Find the longest valid parentheses substring?
3. **Remove minimum**: Minimum removals to make valid?
4. **Generate all**: Generate all valid combinations of n pairs?

---

## 7. Design Hit Counter

**Problem Statement:**
Design a hit counter that counts the number of hits received in the past 5 minutes (300 seconds).

Implement the `HitCounter` class:
- `HitCounter()` Initializes the hit counter
- `void hit(int timestamp)` Records a hit at the given timestamp (in seconds)
- `int getHits(int timestamp)` Returns the number of hits in the past 5 minutes (from `timestamp - 299` to `timestamp` inclusive)

**Constraints:**
- Timestamps are in increasing order (may have duplicates)
- Multiple hits may occur at the same timestamp

**Example:**
```
HitCounter counter = new HitCounter();
counter.hit(1);       // hit at second 1
counter.hit(2);       // hit at second 2
counter.hit(3);       // hit at second 3
counter.getHits(4);   // returns 3 (hits at 1, 2, 3)
counter.hit(300);     // hit at second 300
counter.getHits(300); // returns 4 (hits at 1, 2, 3, 300)
counter.getHits(301); // returns 3 (hits at 2, 3, 300; hit at 1 expired)
```

**Python Solution (Queue-based):**
```python
from collections import deque


class HitCounter:
    """
    Hit counter using a queue to track timestamps.

    Time: O(1) amortized for hit(), O(n) worst case for getHits()
    Space: O(n) where n = number of hits in window
    """

    def __init__(self):
        self.hits: deque[int] = deque()

    def hit(self, timestamp: int) -> None:
        self.hits.append(timestamp)

    def get_hits(self, timestamp: int) -> int:
        # Remove expired hits
        while self.hits and self.hits[0] <= timestamp - 300:
            self.hits.popleft()
        return len(self.hits)


# Test
counter = HitCounter()
counter.hit(1)
counter.hit(2)
counter.hit(3)
print(counter.get_hits(4))    # 3
counter.hit(300)
print(counter.get_hits(300))  # 4
print(counter.get_hits(301))  # 3
```

**Python Solution (Fixed Array - O(1) space):**
```python
class HitCounterOptimized:
    """
    Hit counter using fixed-size arrays for O(1) space.

    Time: O(300) = O(1) for both operations
    Space: O(300) = O(1) fixed
    """

    def __init__(self):
        self.times = [0] * 300   # Timestamp at each bucket
        self.hits = [0] * 300    # Hit count at each bucket

    def hit(self, timestamp: int) -> None:
        idx = timestamp % 300
        if self.times[idx] != timestamp:
            # New time bucket, reset count
            self.times[idx] = timestamp
            self.hits[idx] = 1
        else:
            # Same timestamp, increment
            self.hits[idx] += 1

    def get_hits(self, timestamp: int) -> int:
        total = 0
        for i in range(300):
            if timestamp - self.times[i] < 300:
                total += self.hits[i]
        return total
```

**Complexity Analysis:**

| Approach | hit() | getHits() | Space |
|----------|-------|-----------|-------|
| Queue | O(1) | O(n) amortized | O(n) |
| Fixed Array | O(1) | O(300) = O(1) | O(300) = O(1) |

**Follow-up Questions:**
1. **Concurrency**: How would you handle concurrent hits? (Locks, atomic operations)
2. **Distributed**: How to scale across multiple servers? (Centralized store)
3. **Variable window**: What if the window size is configurable?
4. **High throughput**: What if hit rate is very high? (Batch writes)

---

## References

Sources used for compiling these questions:
- [Plaid Software Engineer Interview Guide - Prepfully](https://prepfully.com/interview-guides/plaid-software-engineer)
- [Plaid Online Assessment Guide - Lodely](https://www.lodely.com/companies/plaid/online-assessment)
- [Plaid Technical Interview Prep - AlgoCademy](https://algocademy.com/blog/plaid-technical-interview-prep-a-comprehensive-guide/)
- [Plaid Interview Questions - Glassdoor](https://www.glassdoor.com/Interview/Plaid-Software-Engineer-Interview-Questions-EI_IE1156368.0,5_KO6,23.htm)
- [Plaid Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/plaid-software-engineer)
