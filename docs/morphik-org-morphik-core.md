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

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Morphik: Unlock the Power of Multimodal Data for Smarter AI Applications

Morphik is an AI-native toolset designed to help developers seamlessly integrate complex context and unstructured data into their AI applications. This repository contains the core components of Morphik, providing a suite of tools for efficient data storage, representation, and retrieval.  Explore the source code on [GitHub](https://github.com/morphik-org/morphik-core).

**Key Features:**

*   **Multimodal Search**: Leverage advanced techniques like ColPali to understand and search visual content within documents, images, PDFs, videos, and more.
*   **Knowledge Graphs**: Build domain-specific knowledge graphs with ease, using pre-built system prompts or custom configurations.
*   **Fast Metadata Extraction**: Quickly extract valuable metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Integrations**: Seamlessly connect with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented Generation**: Enhance generation speed with persistent KV-caches for your documents.

## Why Use Morphik?

Traditional RAG approaches often struggle to handle the complexities of real-world data. Morphik addresses the limitations of these systems by offering a comprehensive solution for managing multimodal data.  This includes:

*   **Avoiding Fragile Pipelines**: Avoid the complexities of integrating disparate tools for text extraction, OCR, embeddings, and vector databases.
*   **Understanding Visually Rich Documents**:  Morphik analyzes charts, diagrams, and tables, capturing all the nuances of visual data.
*   **Improving Accuracy and Performance**: Ensure your AI applications return accurate answers by truly understanding the context within your documents and increasing application speed.
*   **Reducing Infrastructure Costs**: Optimize resource usage by intelligently processing and retrieving information.

## Getting Started

The easiest way to get started with Morphik is by signing up for a free account at [Morphik](https://www.morphik.ai/signup).

## Self-Hosting Morphik

For those who prefer self-hosting, detailed instructions are available [here](https://morphik.ai/docs/getting-started), with options for direct installation and Docker. Please note that we can only provide limited support for self-hosted deployments.

## Using Morphik

Once you've signed up, you can ingest and search data right away.

### Code (Example: Python SDK)

For developers, we offer a Python SDK and a REST API:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Use the web-based console for ingestion, searching, and querying.

### Model Context Protocol

Access Morphik via MCP, with instructions available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions to the project! Please see our [GitHub issues](https://github.com/morphik-org/morphik-core/issues) to report bugs or suggest features and submit pull requests.

## License

Morphik Core is source-available under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise, purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full license text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.