# Samsara Interview Questions

---

## 1. Command Line Arguments Parser

**Problem Statement:**
Design and implement a command line arguments parser. The parser should support the following features:

- **Alias support**: Allow defining an alias (long name) for each argument.
- **Group flags**: Support argument combination, e.g., `-abc` should be equivalent to `-a -b -c`.
- **Help information**: Provide help information for each argument, accessible through `--help`.

Implement the parser as a class with the following methods:

- `add_argument(name, alias, help)`: Add an argument with its alias and help information.
- `parse_args(args)`: Parse a list of arguments and return a dictionary. Keys are argument
  names, values are their parsed values. If no value is specified for an argument, the value
  should be `True` to indicate it is enabled.

**Conventions:**
- A short flag uses a single dash and a single character: `-a`.
- A long alias uses a double dash: `--alias`.
- A value can be attached with `=`: `-a=1` or `--alias=1`.
- Grouped short flags share one dash: `-abc` → `-a -b -c`.
- `--help` is built in; it prints help text and is recorded as `help: True`.
- An unknown flag raises an error.

**Example:**

```
Input:  myprogram -a -b --help
Output: {'a': True, 'b': True, 'help': True}   (and help text is printed)
```

**Test Cases:**

Assume the parser is configured as:
```python
parser.add_argument("a", alias="alias", help="Enable feature A")
parser.add_argument("b", alias="beta",  help="Enable feature B")
parser.add_argument("c", alias="gamma", help="Enable feature C")
```

| Input | Output |
|-------|--------|
| `['myprogram', '-a', '-b', '--help']` | `{'a': True, 'b': True, 'help': True}` (+ prints help) |
| `['myprogram', '-abc']` | `{'a': True, 'b': True, 'c': True}` |
| `['myprogram', '--alias', '-c', '--help']` | `{'a': True, 'c': True, 'help': True}` (+ prints help) |
| `['myprogram', '-a=1', '-b=2']` | `{'a': '1', 'b': '2'}` |
| `['myprogram', '-a', '-unknown']` | raises `ValueError` (unknown flag `-u`) |

**Key Insights:**
1. Store two maps: `name -> metadata` and `alias -> name`, so both `-a` and `--alias` resolve
   to the same canonical name.
2. Classify each token by prefix: `--` is a long alias, a single `-` is one or more short flags.
3. Split on `=` first to separate an optional value from the flag portion.
4. A single-dash token without `=` is treated as a group of short flags; iterate each character.
5. Resolve every flag to a canonical name and raise on anything unknown.

**Python Solution:**
```python
class ArgumentParser:
    """
    A minimal command line argument parser supporting aliases,
    grouped short flags, inline values, and --help.

    Time:  O(k) per parse_args call, where k = total characters in args
    Space: O(a) where a = number of registered arguments
    """

    def __init__(self):
        # name -> {"alias": str | None, "help": str}
        self._args: dict[str, dict] = {}
        # alias -> canonical name
        self._alias_to_name: dict[str, str] = {}

    def add_argument(self, name: str, alias: str | None = None, help: str = "") -> None:
        """Register an argument with an optional long alias and help text."""
        self._args[name] = {"alias": alias, "help": help}
        if alias:
            self._alias_to_name[alias] = name

    def _resolve(self, token: str) -> str | None:
        """Map a short name or a long alias to its canonical name."""
        if token in self._args:
            return token
        return self._alias_to_name.get(token)

    def parse_args(self, args: list[str]) -> dict[str, object]:
        """Parse args (args[0] is the program name) into a dict of values."""
        result: dict[str, object] = {}

        for token in args[1:]:
            if token == "--help":
                self.print_help()
                result["help"] = True
            elif token.startswith("--"):
                # Long alias, optionally with a value: --alias or --alias=val
                key, sep, value = token[2:].partition("=")
                name = self._resolve(key)
                if name is None:
                    raise ValueError(f"Unknown argument: {token}")
                result[name] = value if sep else True
            elif token.startswith("-"):
                body = token[1:]
                if "=" in body:
                    # Short flag with a value: -a=1
                    key, _, value = body.partition("=")
                    name = self._resolve(key)
                    if name is None:
                        raise ValueError(f"Unknown argument: -{key}")
                    result[name] = value
                else:
                    # Grouped short flags: -abc -> -a -b -c
                    for ch in body:
                        name = self._resolve(ch)
                        if name is None:
                            raise ValueError(f"Unknown argument: -{ch}")
                        result[name] = True
            else:
                raise ValueError(f"Unexpected token: {token}")

        return result

    def print_help(self) -> None:
        """Print help information for every registered argument."""
        print("Options:")
        for name, meta in self._args.items():
            alias = f", --{meta['alias']}" if meta["alias"] else ""
            print(f"  -{name}{alias}\t{meta['help']}")


# Example usage
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("a", alias="alias", help="Enable feature A")
    parser.add_argument("b", alias="beta", help="Enable feature B")
    parser.add_argument("c", alias="gamma", help="Enable feature C")

    print(parser.parse_args(["myprogram", "-a", "-b", "--help"]))
    # {'a': True, 'b': True, 'help': True}

    print(parser.parse_args(["myprogram", "-abc"]))
    # {'a': True, 'b': True, 'c': True}

    print(parser.parse_args(["myprogram", "--alias", "-c", "--help"]))
    # {'a': True, 'c': True, 'help': True}

    print(parser.parse_args(["myprogram", "-a=1", "-b=2"]))
    # {'a': '1', 'b': '2'}

    try:
        parser.parse_args(["myprogram", "-a", "-unknown"])
    except ValueError as e:
        print(e)  # Unknown argument: -u
```

**Java Solution:**
```java
import java.util.*;

public class ArgumentParser {

    private static class Meta {
        final String alias;
        final String help;
        Meta(String alias, String help) { this.alias = alias; this.help = help; }
    }

    private final Map<String, Meta> args = new LinkedHashMap<>();
    private final Map<String, String> aliasToName = new HashMap<>();

    public void addArgument(String name, String alias, String help) {
        args.put(name, new Meta(alias, help));
        if (alias != null) {
            aliasToName.put(alias, name);
        }
    }

    private String resolve(String token) {
        if (args.containsKey(token)) return token;
        return aliasToName.get(token);
    }

    public Map<String, Object> parseArgs(String[] argv) {
        Map<String, Object> result = new LinkedHashMap<>();

        for (int i = 1; i < argv.length; i++) {
            String token = argv[i];

            if (token.equals("--help")) {
                printHelp();
                result.put("help", true);
            } else if (token.startsWith("--")) {
                String body = token.substring(2);
                int eq = body.indexOf('=');
                String key = eq >= 0 ? body.substring(0, eq) : body;
                String name = resolve(key);
                if (name == null) throw new IllegalArgumentException("Unknown argument: " + token);
                result.put(name, eq >= 0 ? body.substring(eq + 1) : true);
            } else if (token.startsWith("-")) {
                String body = token.substring(1);
                int eq = body.indexOf('=');
                if (eq >= 0) {
                    String key = body.substring(0, eq);
                    String name = resolve(key);
                    if (name == null) throw new IllegalArgumentException("Unknown argument: -" + key);
                    result.put(name, body.substring(eq + 1));
                } else {
                    for (char ch : body.toCharArray()) {
                        String name = resolve(String.valueOf(ch));
                        if (name == null) throw new IllegalArgumentException("Unknown argument: -" + ch);
                        result.put(name, true);
                    }
                }
            } else {
                throw new IllegalArgumentException("Unexpected token: " + token);
            }
        }
        return result;
    }

    public void printHelp() {
        System.out.println("Options:");
        for (Map.Entry<String, Meta> e : args.entrySet()) {
            String alias = e.getValue().alias != null ? ", --" + e.getValue().alias : "";
            System.out.println("  -" + e.getKey() + alias + "\t" + e.getValue().help);
        }
    }

    public static void main(String[] args) {
        ArgumentParser parser = new ArgumentParser();
        parser.addArgument("a", "alias", "Enable feature A");
        parser.addArgument("b", "beta", "Enable feature B");
        parser.addArgument("c", "gamma", "Enable feature C");

        System.out.println(parser.parseArgs(new String[]{"myprogram", "-abc"}));
        // {a=true, b=true, c=true}
        System.out.println(parser.parseArgs(new String[]{"myprogram", "-a=1", "-b=2"}));
        // {a=1, b=2}
    }
}
```

**TypeScript Solution:**
```typescript
interface Meta {
    alias?: string;
    help: string;
}

class ArgumentParser {
    private args = new Map<string, Meta>();
    private aliasToName = new Map<string, string>();

    addArgument(name: string, alias?: string, help = ""): void {
        this.args.set(name, { alias, help });
        if (alias) {
            this.aliasToName.set(alias, name);
        }
    }

    private resolve(token: string): string | undefined {
        if (this.args.has(token)) return token;
        return this.aliasToName.get(token);
    }

    parseArgs(argv: string[]): Record<string, string | boolean> {
        const result: Record<string, string | boolean> = {};

        for (const token of argv.slice(1)) {
            if (token === "--help") {
                this.printHelp();
                result["help"] = true;
            } else if (token.startsWith("--")) {
                const [key, ...rest] = token.slice(2).split("=");
                const name = this.resolve(key);
                if (!name) throw new Error(`Unknown argument: ${token}`);
                result[name] = rest.length ? rest.join("=") : true;
            } else if (token.startsWith("-")) {
                const body = token.slice(1);
                if (body.includes("=")) {
                    const [key, ...rest] = body.split("=");
                    const name = this.resolve(key);
                    if (!name) throw new Error(`Unknown argument: -${key}`);
                    result[name] = rest.join("=");
                } else {
                    for (const ch of body) {
                        const name = this.resolve(ch);
                        if (!name) throw new Error(`Unknown argument: -${ch}`);
                        result[name] = true;
                    }
                }
            } else {
                throw new Error(`Unexpected token: ${token}`);
            }
        }
        return result;
    }

    printHelp(): void {
        console.log("Options:");
        for (const [name, meta] of this.args) {
            const alias = meta.alias ? `, --${meta.alias}` : "";
            console.log(`  -${name}${alias}\t${meta.help}`);
        }
    }
}

const parser = new ArgumentParser();
parser.addArgument("a", "alias", "Enable feature A");
parser.addArgument("b", "beta", "Enable feature B");
parser.addArgument("c", "gamma", "Enable feature C");

console.log(parser.parseArgs(["myprogram", "-abc"]));       // { a: true, b: true, c: true }
console.log(parser.parseArgs(["myprogram", "-a=1", "-b=2"])); // { a: '1', b: '2' }
```

**Complexity Analysis:**
- **`add_argument`**: O(1) time, O(1) extra space per argument.
- **`parse_args`**: O(k) time, where k = total number of characters across all tokens
  (grouped flags are expanded character by character). O(a) space for the result.

**Edge Cases Handled:**
1. Grouped short flags `-abc` expand to individual flags.
2. Inline values via `=` for both short (`-a=1`) and long (`--alias=1`) forms.
3. Aliases resolve to the same canonical name as their short flag.
4. `--help` prints help text and is recorded as `help: True`.
5. Unknown flags (including unknown characters inside a group) raise `ValueError`.
6. Value strings keep their raw type (`"1"`, not `1`) — no implicit numeric coercion.

**Follow-up Questions:**
1. **Typed values**: How would you support declared types (int, float, bool) with validation?
2. **Required arguments**: How would you enforce that certain arguments must be present?
3. **Repeated flags**: How should `-vvv` (repeat count) or list-valued flags behave?
4. **`--` separator**: How would you treat everything after a bare `--` as positional args?
5. **Ambiguous aliases**: How do you detect and reject duplicate names/aliases at registration?

---

## 2. Convert Text to an HTML Link

**Problem Statement:**
Write a function that receives a string in a specific HTML-like format, e.g.
`Go to Link("whateverlink.com")`, and returns an HTML anchor tag string.

- **Input**: a string, at most 1000 characters, in the format `<label>("<URL>")`.
  The URL is valid and consists of letters, digits, and other URL-legal characters.
- **Output**: the converted HTML link.

The label is the text before `(`, and the URL is the text inside the quotes.

**Example:**

```
Input:  Go to Link("example.com")
Output: <a href="example.com">Go to Link</a>
```

**Requirements:**
Implement it using string splitting and replacement (not a full HTML parser).

**Test Cases:**

| Input | Output |
|-------|--------|
| `Go to Link("whateverlink.com")` | `<a href="whateverlink.com">Go to Link</a>` |
| `Go to Link("example.com")` | `<a href="example.com">Go to Link</a>` |
| `Click Here("https://a.com/x?y=1&z=2")` | `<a href="https://a.com/x?y=1&z=2">Click Here</a>` |

**Key Insights:**
1. Split once on `("` to separate the label from the rest — the label is the left part.
2. Strip the trailing `")` from the right part to recover the raw URL.
3. Format the two pieces into the anchor template. Splitting only once avoids breaking on
   `=`/`&`/`(` characters that may legally appear inside the URL.

**Python Solution:**
```python
def to_html_link(text: str) -> str:
    """
    Convert a string of the form '<label>("<URL>")' into an HTML anchor tag.

    Time:  O(n) where n = len(text)
    Space: O(n) for the output string
    """
    # Split once so URLs containing '("' would not be mis-split (label is unique prefix).
    label, rest = text.split('("', 1)
    # rest looks like:  URL")   -> drop the trailing '")'
    url = rest.rsplit('")', 1)[0]
    return f'<a href="{url}">{label}</a>'


# Example usage
if __name__ == "__main__":
    print(to_html_link('Go to Link("example.com")'))
    # <a href="example.com">Go to Link</a>
    print(to_html_link('Click Here("https://a.com/x?y=1&z=2")'))
    # <a href="https://a.com/x?y=1&z=2">Click Here</a>
```

**Java Solution:**
```java
public class HtmlLink {

    public static String toHtmlLink(String text) {
        // Split once on the '("' delimiter: [label, URL")]
        String[] parts = text.split("\\(\"", 2);
        String label = parts[0];
        String rest = parts[1];
        // Drop the trailing ") from the URL portion
        String url = rest.substring(0, rest.length() - 2);
        return "<a href=\"" + url + "\">" + label + "</a>";
    }

    public static void main(String[] args) {
        System.out.println(toHtmlLink("Go to Link(\"example.com\")"));
        // <a href="example.com">Go to Link</a>
    }
}
```

**TypeScript Solution:**
```typescript
function toHtmlLink(text: string): string {
    // Split once on '("' so URL contents are preserved intact.
    const idx = text.indexOf('("');
    const label = text.slice(0, idx);
    const rest = text.slice(idx + 2);          // URL")
    const url = rest.slice(0, rest.length - 2); // drop trailing ")
    return `<a href="${url}">${label}</a>`;
}

console.log(toHtmlLink('Go to Link("example.com")'));
// <a href="example.com">Go to Link</a>
```

**Complexity Analysis:**
- **Time**: O(n) to scan and slice the string.
- **Space**: O(n) for the constructed output.

**Edge Cases Handled:**
1. URLs containing `=`, `&`, `?`, or `/` are preserved because we split only on the
   `("` delimiter, not on those characters.
2. Labels with internal spaces (`Go to Link`) are kept verbatim.

**Follow-up Questions:**
1. **Validation**: How would you reject malformed inputs missing `("` or the closing `")`?
2. **Escaping**: How would you HTML-escape the label to avoid injection (`<`, `>`, `&`)?
3. **Multiple links**: How would you convert all such patterns within a larger paragraph?
4. **Attributes**: How would you extend the format to carry a title or `target="_blank"`?

---

## 3. Markdown-to-HTML Parser (Phone Screen)

**Problem Statement:**
Build a small markdown processor that converts a plain-text string into valid HTML,
supporting four constructs: paragraphs (`<p>`), soft line breaks (`<br/>`), blockquotes
(`<blockquote>`), and strikethrough (`<del>`). This is a frequently reported Samsara phone
screen; the interviewer typically evolves the requirements mid-problem, so a clean,
extensible scan is more valuable than a clever one-liner.

- **Input**: a string, `0 ≤ length ≤ 200,000` characters.
- **Output**: a valid HTML string with the rendered blocks concatenated in order.

**Rules:**
- **Paragraphs**: consecutive non-blank, non-quoted lines form one `<p>`. A blank line
  (two or more consecutive newlines) separates blocks.
- **Soft line breaks**: a single newline *within* a block becomes `<br/>`.
- **Blockquotes**: consecutive lines beginning with the exact prefix `"> "` form one
  `<blockquote>`. Strip the `"> "` from each line; soft breaks inside are still `<br/>`.
  A blockquote cannot span a paragraph and a switch of block type ends the current block.
  A `>` without the following space (`>not a quote`) is ordinary text.
- **Strikethrough**: text wrapped in a pair of `~~` becomes `<del>…</del>`. Markers pair
  left-to-right and may cross soft line breaks but not block boundaries. An unmatched
  trailing `~~` is literal text.
- **HTML escaping**: escape text content — `&`→`&amp;`, `<`→`&lt;`, `>`→`&gt;` — before
  inserting any generated tags.

**Example:**

```
Input:
Hello ~~world~~
> quoted line
> second quote

New paragraph

Output:
<p>Hello <del>world</del></p><blockquote>quoted line<br/>second quote</blockquote><p>New paragraph</p>
```

**Key Insights:**
1. **Two levels**: first split into *blocks* (paragraph vs. blockquote, delimited by blank
   lines and type switches), then render inline formatting within each block.
2. **Escape before tagging**: escape `&<>` on the raw text first so your own `<del>`,
   `<br/>`, `<p>` tags survive; tildes and newlines pass through escaping untouched.
3. **Strikethrough scan**: only open a `<del>` if a later `~~` exists in the same block,
   otherwise emit the tildes literally — this handles unmatched and adjacent (`~~~~`) cases.
4. Join a block's content with `\n`, run escape → strikethrough → replace `\n` with `<br/>`.

**Python Solution:**
```python
def markdown_to_html(text: str) -> str:
    """
    Convert markdown-ish text to HTML (p, br, blockquote, del).

    Time:  O(n) where n = len(text)
    Space: O(n) for the output
    """
    def escape(s: str) -> str:
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

    def strikethrough(s: str) -> str:
        out, i, open_del = [], 0, False
        while i < len(s):
            if s[i:i + 2] == "~~":
                if not open_del and s.find("~~", i + 2) != -1:
                    out.append("<del>"); open_del = True
                elif open_del:
                    out.append("</del>"); open_del = False
                else:
                    out.append("~~")  # unmatched -> literal
                i += 2
            else:
                out.append(s[i]); i += 1
        return "".join(out)

    # Group lines into blocks: (type, [content_lines])
    blocks: list[tuple[str, list[str]]] = []
    cur_type: str | None = None
    cur_lines: list[str] = []

    def flush():
        if cur_lines:
            blocks.append((cur_type, list(cur_lines)))

    for line in text.split("\n"):
        if line.strip() == "":            # blank line ends the block
            flush(); cur_type, cur_lines = None, []
            continue
        if line.startswith("> "):
            t, content = "quote", line[2:]
        else:
            t, content = "para", line
        if t != cur_type:                 # type switch ends the block
            flush(); cur_type, cur_lines = t, []
        cur_lines.append(content)
    flush()

    html = []
    for btype, lines in blocks:
        inner = strikethrough(escape("\n".join(lines))).replace("\n", "<br/>")
        tag = "blockquote" if btype == "quote" else "p"
        html.append(f"<{tag}>{inner}</{tag}>")
    return "".join(html)


# Example usage
if __name__ == "__main__":
    src = "Hello ~~world~~\n> quoted line\n> second quote\n\nNew paragraph"
    print(markdown_to_html(src))
    # <p>Hello <del>world</del></p><blockquote>quoted line<br/>second quote</blockquote><p>New paragraph</p>
```

**Java Solution:**
```java
import java.util.*;

public class MarkdownToHtml {

    private static String escape(String s) {
        return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
    }

    private static String strikethrough(String s) {
        StringBuilder out = new StringBuilder();
        int i = 0;
        boolean openDel = false;
        while (i < s.length()) {
            if (i + 1 < s.length() && s.charAt(i) == '~' && s.charAt(i + 1) == '~') {
                if (!openDel && s.indexOf("~~", i + 2) != -1) {
                    out.append("<del>"); openDel = true;
                } else if (openDel) {
                    out.append("</del>"); openDel = false;
                } else {
                    out.append("~~");
                }
                i += 2;
            } else {
                out.append(s.charAt(i));
                i++;
            }
        }
        return out.toString();
    }

    public static String markdownToHtml(String text) {
        List<String[]> blocks = new ArrayList<>();   // {type, joinedContent}
        String curType = null;
        List<String> curLines = new ArrayList<>();

        for (String line : text.split("\n", -1)) {
            if (line.trim().isEmpty()) {
                if (!curLines.isEmpty()) {
                    blocks.add(new String[]{curType, String.join("\n", curLines)});
                }
                curType = null; curLines = new ArrayList<>();
                continue;
            }
            String t, content;
            if (line.startsWith("> ")) { t = "quote"; content = line.substring(2); }
            else { t = "para"; content = line; }
            if (!t.equals(curType)) {
                if (!curLines.isEmpty()) {
                    blocks.add(new String[]{curType, String.join("\n", curLines)});
                }
                curType = t; curLines = new ArrayList<>();
            }
            curLines.add(content);
        }
        if (!curLines.isEmpty()) {
            blocks.add(new String[]{curType, String.join("\n", curLines)});
        }

        StringBuilder html = new StringBuilder();
        for (String[] b : blocks) {
            String inner = strikethrough(escape(b[1])).replace("\n", "<br/>");
            String tag = "quote".equals(b[0]) ? "blockquote" : "p";
            html.append("<").append(tag).append(">")
                .append(inner)
                .append("</").append(tag).append(">");
        }
        return html.toString();
    }
}
```

**Complexity Analysis:**
- **Time**: O(n) — each character is scanned a constant number of times (the `find`/`indexOf`
  look-ahead for `~~` is bounded overall since we advance past matched pairs).
- **Space**: O(n) for the block buffers and output string.

**Edge Cases Handled:**
1. `>not a quote` (no space) → treated as a paragraph, `>` escaped to `&gt;`.
2. Consecutive blockquote lines collapse into one `<blockquote>`; a following text line
   starts a new `<p>`.
3. `~~~~` → empty `<del></del>`; a lone unmatched `~~` stays literal.
4. Blank-only / whitespace-only input → empty output.
5. Strikethrough spanning a soft line break inside one block works; it cannot cross a
   blank line into the next block.
6. Raw `&`, `<`, `>` in text are escaped without touching generated tags.

**Follow-up Questions:**
1. **More constructs**: How would you add `**bold**`, `*italic*`, or headings without a rewrite?
2. **Nested formatting**: How would you support `~~a **b** c~~` with a proper parse tree?
3. **Streaming**: How would you emit HTML incrementally for a 200 KB input?
4. **Malformed input**: How do you keep output well-formed when tags are unbalanced?

---

## 4. Implement `strStr` (Substring Index)

**Problem Statement:**
Reported as a warm-up/phone question: given two strings `haystack` and `needle`, return the
index of the first occurrence of `needle` in `haystack`, or `-1` if it is not present. An
empty `needle` returns `0`.

**Examples:**

| haystack | needle | Output |
|----------|--------|--------|
| `"sadbutsad"` | `"sad"` | `0` |
| `"leetcode"` | `"leeto"` | `-1` |
| `"hello"` | `""` | `0` |
| `"abc"` | `"c"` | `2` |

**Key Insights:**
1. A simple sliding window compares `needle` against each length-`m` window of `haystack`;
   O(n·m) worst case but often fine for interview inputs.
2. For linear time, KMP precomputes the longest-proper-prefix-suffix (LPS) array so the scan
   never backtracks in `haystack`, giving O(n + m).

**Python Solution (KMP, O(n + m)):**
```python
def str_str(haystack: str, needle: str) -> int:
    """Return first index of needle in haystack, or -1. O(n + m) time, O(m) space."""
    if needle == "":
        return 0
    m = len(needle)

    # Build LPS: lps[i] = length of longest proper prefix of needle[:i+1]
    # that is also a suffix.
    lps = [0] * m
    length = 0
    for i in range(1, m):
        while length > 0 and needle[i] != needle[length]:
            length = lps[length - 1]
        if needle[i] == needle[length]:
            length += 1
        lps[i] = length

    # Scan haystack without backtracking.
    j = 0
    for i, ch in enumerate(haystack):
        while j > 0 and ch != needle[j]:
            j = lps[j - 1]
        if ch == needle[j]:
            j += 1
        if j == m:
            return i - m + 1
    return -1


# Test cases
print(str_str("sadbutsad", "sad"))  # 0
print(str_str("leetcode", "leeto")) # -1
print(str_str("hello", ""))         # 0
print(str_str("abc", "c"))          # 2
```

**Java Solution:**
```java
public int strStr(String haystack, String needle) {
    if (needle.isEmpty()) return 0;
    int m = needle.length();

    int[] lps = new int[m];
    int len = 0;
    for (int i = 1; i < m; i++) {
        while (len > 0 && needle.charAt(i) != needle.charAt(len)) {
            len = lps[len - 1];
        }
        if (needle.charAt(i) == needle.charAt(len)) len++;
        lps[i] = len;
    }

    int j = 0;
    for (int i = 0; i < haystack.length(); i++) {
        char ch = haystack.charAt(i);
        while (j > 0 && ch != needle.charAt(j)) j = lps[j - 1];
        if (ch == needle.charAt(j)) j++;
        if (j == m) return i - m + 1;
    }
    return -1;
}
```

**TypeScript Solution:**
```typescript
function strStr(haystack: string, needle: string): number {
    if (needle === "") return 0;
    const m = needle.length;

    const lps = new Array(m).fill(0);
    let len = 0;
    for (let i = 1; i < m; i++) {
        while (len > 0 && needle[i] !== needle[len]) len = lps[len - 1];
        if (needle[i] === needle[len]) len++;
        lps[i] = len;
    }

    let j = 0;
    for (let i = 0; i < haystack.length; i++) {
        while (j > 0 && haystack[i] !== needle[j]) j = lps[j - 1];
        if (haystack[i] === needle[j]) j++;
        if (j === m) return i - m + 1;
    }
    return -1;
}
```

**Complexity Analysis:**
- **Sliding window**: O(n·m) time, O(1) space.
- **KMP**: O(n + m) time, O(m) space for the LPS array.

**Follow-up Questions:**
1. **All occurrences**: Return every start index, not just the first.
2. **Case-insensitive / Unicode**: How do normalization and locale affect matching?
3. **Multiple patterns**: How would you match many needles at once (Aho–Corasick)?
4. **Streaming haystack**: How would you match against a stream you can't fully buffer?

---

## 5. System Design — Fleet Fuel-Card Payment Tracking

**Problem Statement:**
Reported onsite system-design round. Design a service where fleet operators upload CSV files
of card payments for refueling vehicles across a fleet, and the system ingests, validates,
deduplicates, and makes the payments queryable for tracking and reconciliation.

**Functional Requirements:**
- Operators upload CSV files (each row = one fuel transaction: card id, vehicle id,
  driver id, timestamp, gallons, amount, merchant/location).
- Ingest and validate rows; reject/flag malformed rows without failing the whole file.
- Deduplicate transactions (same file re-uploaded, or overlapping exports).
- Query/track spend by fleet, vehicle, driver, card, and time range.
- Surface anomalies (e.g., amount far above vehicle's tank capacity).

**Non-Functional Requirements:**
- Files range from a few KB to hundreds of MB; ingestion should be async and resumable.
- Reads (dashboards, reports) should be fast and eventually consistent with ingestion.
- Idempotent uploads; a retried upload must not double-count.
- Auditability: keep the raw file and the parse/validation outcome per row.

**High-Level Design:**
1. **Upload**: client requests a pre-signed URL and uploads the CSV directly to object
   storage (S3/GCS). This offloads large-file transfer from the API tier.
2. **Ingestion trigger**: the upload emits an event (S3 notification → queue). A worker pool
   consumes the queue so ingestion scales horizontally and is retryable.
3. **Parse & validate worker**: streams the file row-by-row (no full load into memory),
   validates each row, computes a deterministic `transaction_id` (see idempotency), and
   writes valid rows to the transactions store; invalid rows go to a dead-letter/errors table
   with the reason and line number.
4. **Storage**:
   - **Raw files** in object storage (immutable, for audit/replay).
   - **Transactions** in a relational/columnar store partitioned by `fleet_id` and time,
     indexed on `(fleet_id, vehicle_id, ts)` and `(card_id, ts)` for common queries.
   - **File/ingestion metadata** (status, row counts, errors) in a small table.
5. **Query API / read layer**: serves dashboards; heavy aggregations (spend per vehicle per
   month) are pre-aggregated via a rollup job or materialized views.

**Idempotency & Deduplication:**
- Derive `transaction_id = hash(card_id, timestamp, amount, merchant, vehicle_id)` so the
  same logical transaction always maps to the same key; insert with "on conflict do nothing".
- Track processed files by content hash to short-circuit exact re-uploads.

**Data Model (sketch):**
```
files(file_id, fleet_id, object_key, content_hash, status, total_rows,
      valid_rows, error_rows, uploaded_at)
transactions(transaction_id PK, fleet_id, vehicle_id, driver_id, card_id,
             ts, gallons, amount, merchant, file_id)
row_errors(file_id, line_no, raw_line, reason)
```

**Scaling & Reliability:**
- Queue + stateless workers → scale with file volume; backpressure via queue depth.
- Partition transactions by `fleet_id`/time to keep queries and retention manageable.
- At-least-once processing made safe by the deterministic `transaction_id` upsert.
- Anomaly detection can run as a downstream consumer on newly ingested rows.

**Follow-up Questions:**
1. **Schema drift**: How do you handle different CSV column layouts per operator/provider?
2. **Partial failure**: A worker dies mid-file — how do you resume without duplicates?
3. **Real-time**: How would you add streaming (per-swipe) ingestion alongside batch CSV?
4. **Reconciliation**: How do you reconcile against the card processor's authoritative ledger?

---

## References

Sources used for compiling these questions:
- [Samsara Software Engineer Interview Questions - Glassdoor](https://www.glassdoor.com/Interview/Samsara-Software-Engineer-Interview-Questions-EI_IE1169265.0,7_KO8,25.htm)
- [Samsara Software Engineer Interview Guide - InterviewQuery](https://www.interviewquery.com/interview-guides/samsara-software-engineer)
- [Samsara | Phone interview | Parse the String - LeetCode Discuss](https://leetcode.com/discuss/interview-question/1034881/samsara-telephonic-interview-problem)
- [Samsara | Phone | Convert Markdown string to HTML - LeetCode Discuss](https://leetcode.com/discuss/interview-question/948848/samsara-phone-convert-markdown-string-to-html/)
- [Implement Markdown-to-HTML parser - Samsara Questions (Prachub)](https://prachub.com/interview-questions/implement-markdown-to-html-parser)
- [Samsara Interview Questions and Process - AlgoDaily](https://algodaily.com/companies/samsara)
- [Samsara interview question bank - Prepfully](https://prepfully.com/interview-questions/samsara)
- Common CLI parser and `strStr` design exercises (argparse-style / LeetCode #28)
