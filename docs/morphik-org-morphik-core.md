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

<p align="center">
  <a href="https://morphik.ai/docs">Docs</a> - <a href="https://discord.gg/BwMtv3Zaju">Community</a> - <a href="https://morphik.ai/docs/blogs/gpt-vs-morphik-multimodal">Why Morphik?</a> - <a href="https://github.com/morphik-org/morphik-core/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

## Morphik: Unlock the Power of Your Data with AI-Native Document Understanding

Tired of traditional RAG pipelines that fall short?  **Morphik** ([original repo](https://github.com/morphik-org/morphik-core)) provides developers with the tools to build AI applications that truly understand complex, unstructured, and multimodal data.

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

*   **Multimodal Search**: Go beyond text and search over images, PDFs, videos, and more, understanding the visual content of your documents.
*   **Knowledge Graphs**: Build domain-specific knowledge graphs with ease, using pre-built system prompts or customizing your own.
*   **Fast and Scalable Metadata Extraction**: Efficiently extract valuable metadata, including bounding boxes, labels, and classifications.
*   **Integrations**: Seamlessly integrate with your existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation**: Accelerate generation with persistent KV-caches of your documents.

## Why Morphik?

Building AI applications that interact with data shouldn't be a complex endeavor. Morphik simplifies the process by providing a unified solution for ingesting, searching, transforming, and managing unstructured and multimodal documents.  It overcomes the limitations of traditional RAG approaches, which often struggle with visually rich documents and become fragile at scale. With Morphik, your AI applications can finally understand the full context of your data, leading to more accurate and reliable results.

## Getting Started

### Free Tier
The easiest way to start is by signing up for free at [Morphik](https://www.morphik.ai/signup).

### Self-Hosting
For self-hosting, see [Morphik Documentation](https://morphik.ai/docs/getting-started).

**Important**: Due to limited resources, we cannot provide full support for self-hosted deployments. We have an installation guide, and a [Discord community](https://discord.gg/BwMtv3Zaju) to help, but we can't guarantee full support.

## Using Morphik

### Code (Example: Python SDK)

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Interact with Morphik via a web-based interface to ingest, search, and query your data.

### Model Context Protocol

Access Morphik via MCP, with instructions available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions!  Please feel free to:

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submit feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Create pull requests

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

-   **Personal / Indie use**: free.
-   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
-   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.