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

# Morphik: Unlock the Power of Multimodal Data for AI Applications

Morphik is your all-in-one solution for building AI applications that understand and interact with visually rich documents and multimodal data.  [Explore the Morphik Core on GitHub](https://github.com/morphik-org/morphik-core).

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

*   **Multimodal Search:**  Go beyond text-based search.  Morphik understands the *visual* content of your documents, allowing you to search images, PDFs, videos, and more with a single endpoint.
*   **Knowledge Graphs:** Easily build knowledge graphs for domain-specific use cases with a single line of code. Leverage pre-built system prompts or bring your own.
*   **Fast and Scalable Metadata Extraction:** Extract valuable metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Integrations:**  Seamlessly integrate with your existing tools and workflows.  Currently offering integrations with (but not limited to) Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:**  Create persistent KV-caches of your documents to significantly speed up generation and reduce infrastructure costs.

## The Problem: Why Morphik?

Traditional approaches to building AI applications often struggle with the complexities of real-world data.  Morphik addresses the limitations of fragile, multi-tool pipelines that fail to handle:

*   **Visually Rich Documents:**  Charts, diagrams, and tables are often misinterpreted or lost entirely by traditional methods.
*   **Data Siloing:**  Cobbling together separate systems for data extraction, embedding, and retrieval creates brittle systems that are hard to maintain.
*   **Scalability and Cost:**  Relying on inefficient pipelines leads to exploding infrastructure costs.

## Solution:  How Morphik Helps

Morphik offers a comprehensive toolset to streamline your AI application development:

*   **Simplified Data Handling:** Ingest, search, transform, and manage unstructured and multimodal documents with ease.
*   **Improved Accuracy:**  Gain a deeper understanding of your data, leading to more accurate AI responses.
*   **Cost Optimization:**  Reduce infrastructure costs by leveraging efficient caching and retrieval mechanisms.

## Getting Started

The easiest way to get started is by signing up for a free account at [Morphik](https://www.morphik.ai/signup).

## Table of Contents
- [Getting Started with Morphik](#getting-started-with-morphik-recommended)
- [Self-hosting Morphik](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [Open source vs paid](#License)

## Getting Started with Morphik (Recommended)

Sign up for free at [Morphik](https://www.morphik.ai/signup) to start using Morphik. We offer a generous free tier.

## Self-hosting Morphik

For self-hosting instructions, see [here](https://morphik.ai/docs/getting-started).

**Important**: Due to limited resources, we cannot provide full support for self-hosted deployments. We have an installation guide, and a [Discord community](https://discord.gg/BwMtv3Zaju) to help, but we can't guarantee full support.

## Using Morphik

### Code (Example: Python SDK)
For programmers, we offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check). Ingesting a file is as simple as:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Similarly, searching and querying your data is easy too:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

You can also interact with Morphik via the Morphik Console. This is a web-based interface that allows you to ingest, search, and query your data. You can upload files, connect to different data sources, and chat with your data all within the same place.

### Model Context Protocol

Finally, you can also access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions!  We especially encourage:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

-   **Personal / Indie use**: free.
-   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
-   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.