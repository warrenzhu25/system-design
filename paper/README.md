# Database Systems Paper Notes

This directory contains readable Markdown notes for influential industrial database-system
papers. Each note links to the authoritative publication and clearly identifies its venue and
award status.

## Collection

| System or paper | Venue and status | Markdown |
| --- | --- | --- |
| Ursa | VLDB 2025 Best Industry Paper | [Reading notes](vldb-2025-best-industry-ursa.md) |

## Suggested next papers

The systems below are related to the requested high-performance database theme, but they are
not all VLDB Best Industry Papers. Keeping that distinction explicit avoids attributing an
award or venue to the wrong work.

| Topic | Best primary reading | Publication status |
| --- | --- | --- |
| Dragonfly | [Dragonfly architecture documentation](https://www.dragonflydb.io/docs) | Project documentation |
| Dashtable | [Dragonfly cache design](https://www.dragonflydb.io/blog/dragonfly-cache-design) | Engineering article based on the original Dash hashing work |
| Seastar | [Seastar documentation](https://seastar.io/) | Project documentation |
| VLL | [VLL: A Lock Manager Redesign for Main Memory Database Systems](https://www.vldb.org/vldb_journal/index.php?id=1342&option=com_article_manager&view=article) | VLDB Journal 24(5), 2015 |
| ScyllaDB | [ScyllaDB architecture documentation](https://opensource.docs.scylladb.com/stable/architecture/index.html) | Project documentation |
| DataFusion | [Apache Arrow DataFusion: A Fast, Embeddable, Modular Analytic Query Engine](https://doi.org/10.1145/3626246.3653368) | SIGMOD 2024 Industrial Track |

## Note format

Each Markdown file should contain:

1. Complete citation, DOI, venue, award, and primary-source links.
2. The problem and design goals.
3. Architecture and important data flows.
4. Correctness model and failure behavior.
5. Evaluation results, with the authors' assumptions preserved.
6. Limitations, open questions, and practical lessons.

The files are study notes, not substitutes for the published papers. Claims and benchmark
numbers should be checked against the linked source before being used in production decisions.
