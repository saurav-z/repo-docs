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

## Morphik Core: Unlock the Power of Multimodal Data for Your AI Applications

Morphik Core is an open-source toolkit designed to revolutionize how developers integrate complex data into their AI applications, going beyond traditional RAG limitations to provide truly comprehensive data understanding.  **[Explore the original repository on GitHub](https://github.com/morphik-org/morphik-core).**

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

*   **Multimodal Search:** Leverage advanced techniques like ColPali to search across images, PDFs, videos, and more, truly understanding visual content.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs in a single line of code, using battle-tested system prompts or your own.
*   **Fast and Scalable Metadata Extraction:** Efficiently extract metadata from documents, including bounding boxes, labeling, and classification.
*   **Integrations:** Seamlessly integrate with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Speed up generation by creating persistent KV-caches of your documents.

## Why Morphik?

Tired of AI applications that misunderstand visually rich documents and struggle with complex data? Morphik offers a comprehensive solution to the limitations of traditional RAG approaches. It helps you build AI applications that:

*   **Understand Visual Data:** Go beyond simple text extraction to truly understand images, charts, diagrams, and other visual elements.
*   **Maintain Accuracy:** Avoid the pitfalls of misinterpreting technical specifications and complex documents.
*   **Optimize Performance & Reduce Costs:** Reduce infrastructure costs and improve query performance by efficiently processing and retrieving data.

## Getting Started

### Recommended: Morphik Cloud (Free Tier)

The easiest way to get started is to sign up for a free account at [Morphik](https://www.morphik.ai/signup). We offer a generous free tier and transparent, compute-usage-based pricing.

### Self-hosting (Open Source)

For those who prefer self-hosting, find the dedicated installation instructions [here](https://morphik.ai/docs/getting-started).  Installation options include direct installation and Docker.

**Important:**  Due to limited resources, support for self-hosted deployments is limited. The [Discord community](https://discord.gg/BwMtv3Zaju) is available for assistance.

## Using Morphik

Once you have access to Morphik, you can quickly ingest and search your data using the following methods:

### Code (Python SDK)

For developers, the [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and [REST API](https://morphik.ai/docs/api-reference/health-check) offer seamless integration:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

query_result = morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Use the web-based [Morphik Console](<>) to ingest, search, and query your data with a user-friendly interface.  Upload files, connect to data sources, and interact with your data all in one place.

### Model Context Protocol (MCP)

Access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions to Morphik Core!  Help us improve the project by:

*   Reporting bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggesting new features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submitting pull requests

We're currently focused on:

*   Improving performance.
*   Expanding integrations with other tools.
*   Identifying and incorporating valuable research papers.

Share your ideas and feedback on Discord or GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use:** Free.
*   **Commercial production use:** Free if your Morphik deployment generates < $2,000/month in gross revenue. Otherwise, purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source:** Each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full license text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.