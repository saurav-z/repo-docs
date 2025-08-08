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

## Morphik: Unlock the Power of Your Documents with AI

Morphik is an AI-native toolset for developers, designed to seamlessly integrate complex context into AI applications, offering a robust solution for managing and searching unstructured and multimodal data. [Explore the Morphik Core on GitHub](https://github.com/morphik-org/morphik-core).

> **Important Migration Note:** If you installed Morphik before June 22nd, 2025, enhance your query performance by running the migration script:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

*   **Multimodal Search:** Go beyond text with search that understands the visual content of your documents, including images, PDFs, and videos, powered by techniques like ColPali.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs quickly and easily with a single line of code, utilizing our pre-built system prompts or your own.
*   **Fast Metadata Extraction:** Extract crucial metadata from your documents, including bounding boxes, labels, and classifications, efficiently and at scale.
*   **Integrations:** Seamlessly integrate with tools like Google Suite, Slack, and Confluence, expanding the reach of your data processing.
*   **Cache-Augmented Generation:** Improve efficiency and speed up generation with persistent KV-caches for your documents.

## Why Morphik?

Traditional RAG approaches often struggle in production, especially when handling visually rich documents.  Morphik addresses the shortcomings of traditional approaches by providing a unified solution that:

*   **Understands Visual Data:**  Accurately interprets charts, diagrams, and other visual elements, delivering more precise results.
*   **Simplifies Pipelines:**  Eliminates the need to juggle multiple tools for data processing, reducing complexity and maintenance.
*   **Enhances Performance:**  Optimizes data ingestion and retrieval, leading to faster query times and reduced infrastructure costs.

## Getting Started

Morphik offers a [free tier](https://www.morphik.ai/pricing) for developers.

1.  **Sign Up:** Get started by signing up at [Morphik](https://www.morphik.ai/signup).
2.  **Explore the Docs:** Dive deeper into the features and capabilities with our detailed documentation at [https://morphik.ai/docs](https://morphik.ai/docs).

## Usage

### Code Example (Python SDK)

Here's a quick example of how to ingest and search data using the Python SDK:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Search your data:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Interact with Morphik through a web-based interface for easy file ingestion, search, and data querying.

### Model Context Protocol

Access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Self-Hosting (Open Source)

For those who prefer self-hosting, detailed instructions are available [here](https://morphik.ai/docs/getting-started).  Note: We can't provide full support for self-hosted deployments. Use the installation guide and the [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

## Contributing

Contributions are welcome! We encourage:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

Our current focus is on improving performance, integrating with more tools, and finding valuable research papers to improve user experience.  Share your thoughts in the Discord or on GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.