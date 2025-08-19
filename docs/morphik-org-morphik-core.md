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

## Morphik: Unlock the Power of Your Multimodal Data with AI

Morphik is the AI-native platform that empowers developers to seamlessly integrate context and derive insights from visually rich documents and multimodal data.  Explore the [Morphik Core GitHub Repository](https://github.com/morphik-org/morphik-core).

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

*   **Multimodal Search:** Leverage techniques like ColPali to search over images, PDFs, videos, and more with a single endpoint, understanding the visual content within.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs effortlessly with a single line of code, using our battle-tested system prompts or your own.
*   **Fast and Scalable Metadata Extraction:** Quickly extract metadata from documents, including bounding boxes, labeling, classification, and more.
*   **Integrations:** Seamlessly integrate with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Utilize persistent KV-caches to accelerate generation.

## Why Choose Morphik?

Tired of piecing together disparate tools for AI applications? Morphik simplifies the process, addressing the limitations of traditional RAG approaches, especially when dealing with visually rich documents.  Avoid the pitfalls of fragile pipelines and ensure your AI applications truly understand and utilize all your data.

## Getting Started

### Recommended: Morphik Cloud

The easiest way to get started is by signing up for a free account at [Morphik](https://www.morphik.ai/signup). We offer a generous free tier and transparent, compute-usage based pricing.

### Self-hosting Morphik

For self-hosting, refer to the dedicated instructions [here](https://morphik.ai/docs/getting-started).  We offer both direct installation and Docker options. *Note: full support for self-hosted deployments is limited.*

## Using Morphik

### Code (Python SDK Example)

Ingest and query your data effortlessly using our Python SDK:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Alternatively, use the web-based Morphik Console for intuitive ingestion, search, and querying:

### Model Context Protocol

Morphik is also accessible via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contribute

We welcome contributions!

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Suggest features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Submit pull requests.

We're currently focused on improving speed, expanding integrations, and researching valuable papers. Share your thoughts on Discord or GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.