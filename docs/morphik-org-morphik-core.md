<p align="center">
  <img alt="Morphik Logo" src="assets/morphik_logo.png">
</p>
<p align="center">
  <a href='http://makeapullrequest.com'><img alt='PRs Welcome' src='https://img.shields.io/badge/PRs-welcome-brightgreen.svg?style=shields'/></a>
  <img alt="GitHub commit activity" src="https://img.shields.io/github/commit-activity/m/morphik-org/morphik-core"/>
  <img alt="GitHub closed issues" src="https://img.shields.io/github/issues-closed/morphik-org/morphik-core"/>
  <img alt="PyPI - Downloads" src="https://img.shields.io/pypi/dm/morphik">
  <a href="https://discord.gg/BwMtv3Zaju"><img alt="Discord" src="https://img.shields.io/discord/1336524712817332276?logo=discord&label=discord"></a>
</p>

<!-- add a roadmap! - <a href="https://morphik.ai/roadmap">Roadmap</a> - -->
<!-- Add a changelog! - <a href="https://morphik.ai/changelog">Changelog</a> -->

<p align="center">
  <a href="https://morphik.ai/docs">Docs</a> - <a href="https://discord.gg/BwMtv3Zaju">Community</a> - <a href="https://morphik.ai/docs/blogs/gpt-vs-morphik-multimodal">Why Morphik?</a> - <a href="https://github.com/morphik-org/morphik-core/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

## Morphik: Unlock the Power of Multimodal Data with AI-Powered Document Understanding

Morphik is a powerful AI-native toolset designed for developers to seamlessly integrate complex context and uncover insights from visually rich documents and multimodal data.  Explore the core functionality and capabilities of the open-source project on [GitHub](https://github.com/morphik-org/morphik-core).

**Important Note for Existing Installations:** If you installed Morphik before June 22nd, 2025, please run the migration script for faster query performance:
```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

### Key Features:

*   **Multimodal Search:**  Go beyond text-based search and understand the visual content of your documents. Search images, PDFs, videos, and more with a single endpoint, leveraging techniques like ColPali.
*   **Knowledge Graphs:**  Easily build domain-specific knowledge graphs with a single line of code. Utilize battle-tested system prompts or customize your own.
*   **Fast and Scalable Metadata Extraction:**  Efficiently extract key metadata from documents, including bounding boxes, labels, and classifications.
*   **Integrations:** Connect Morphik with existing tools and workflows, including integrations with Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:**  Create persistent KV-caches of your documents to speed up generation.
*   **Free Tier:** Get started today with Morphik's generous free tier available at [Morphik](https://www.morphik.ai/signup).

### Why Morphik?

Traditional RAG approaches often struggle with real-world data, especially visually rich documents.  Morphik addresses these limitations by:

*   **Overcoming Fragile Pipelines:** Eliminates the need to duct-tape together multiple, often incompatible, tools.
*   **Understanding Visual Data:**  Accurately interprets charts, diagrams, and other visual elements often missed by standard approaches.
*   **Improving Performance:** Reduces infrastructure costs by avoiding redundant processing of the same information.

### Getting Started

The easiest way to get started is by signing up for free at [Morphik](https://www.morphik.ai/signup).

### Self-Hosting

For self-hosting instructions, please see our documentation [here](https://morphik.ai/docs/getting-started).
*Note:* Due to limited resources, we cannot provide full support for self-hosted deployments.

### Using Morphik

Once you've signed up, you can begin ingesting and searching your data immediately.

#### Code Example (Python SDK):

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

#### Searching and Querying:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

#### Morphik Console

Interact with Morphik through the web-based console for data ingestion, searching, and querying.

#### Model Context Protocol (MCP)

Access Morphik via MCP following the instructions [here](https://morphik.ai/docs/using-morphik/mcp).

### Contributing

We welcome contributions!  Please submit:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

We are currently focused on improving performance, expanding integrations, and identifying valuable research papers.

### License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

### Contributors

Thank you to all our contributors!  See our [special thanks page](https://morphik.ai/docs/special-thanks) for more information.