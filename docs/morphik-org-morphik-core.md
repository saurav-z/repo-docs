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

# Morphik: The AI-Native Toolkit for Intelligent Document Understanding

Morphik empowers developers to build AI applications that truly understand and interact with complex, multimodal data.  For the open-source core, visit the [Morphik Core GitHub Repository](https://github.com/morphik-org/morphik-core).

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

Morphik provides a comprehensive suite of tools for advanced document processing and retrieval:

*   **Multimodal Search:**  Unleash the power of search that understands visual content like images, PDFs, and videos.
*   **Knowledge Graphs:**  Easily build domain-specific knowledge graphs with a single line of code.
*   **Fast Metadata Extraction:**  Rapidly extract metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Integrations:** Seamlessly integrate with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented Generation:**  Boost performance and reduce costs with persistent KV-caches of your documents.

## Why Morphik?

Tired of struggling with brittle RAG pipelines that fail to handle real-world complexity?  Morphik offers an end-to-end solution that addresses the shortcomings of traditional approaches, enabling you to build AI applications that can truly understand and leverage the information within your documents, no matter how visually rich.

## Getting Started

The easiest way to get started is by signing up for a [free Morphik account](https://www.morphik.ai/signup). Explore the documentation for [detailed tutorials](https://morphik.ai/docs).

## Self-hosting the Open-Source Version

For those who prefer self-hosting, follow the instructions [here](https://morphik.ai/docs/getting-started).

**Important**:  Full support for self-hosted deployments is limited.  Consult the [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

## Using Morphik

### Code (Example: Python SDK)

Ingesting and searching data with the Python SDK is straightforward:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

# Search your data
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

The Morphik Console provides a web-based interface for data ingestion, searching, and querying.

### Model Context Protocol

Access Morphik via the Model Context Protocol (MCP) using the instructions [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome your contributions! Please submit:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.