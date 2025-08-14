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

## Morphik: Unlock the Power of Multimodal Data with AI-Native Search

Morphik provides developers with a powerful, AI-native toolset for visually rich documents and multimodal data, enabling more accurate and insightful AI applications. Learn more on the [Morphik Core GitHub](https://github.com/morphik-org/morphik-core).

**Key Features:**

*   **Multimodal Search:**  Go beyond text-based search with our advanced ColPali technology, allowing you to search images, PDFs, videos, and more, truly understanding the visual content.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs quickly and easily with a single line of code, utilizing our pre-built system prompts or creating your own.
*   **Fast Metadata Extraction:**  Efficiently extract valuable metadata from your documents, including bounding boxes, labels, and classifications, with speed and scalability in mind.
*   **Seamless Integrations:**  Connect Morphik with your existing tools and workflows, including popular platforms like Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Boost performance with persistent KV-caches of your documents, accelerating the generation process.

Get started today with a [free tier](https://www.morphik.ai/signup) and revolutionize how you interact with your data!

## Table of Contents
- [Getting Started with Morphik](#getting-started-with-morphik-recommended)
- [Self-hosting Morphik](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [License](#License)

## Getting Started with Morphik (Recommended)

The easiest way to get started with Morphik is by signing up for free at [Morphik](https://www.morphik.ai/signup). We offer a generous free tier and transparent, compute-usage based pricing for larger data ingestion needs.

## Self-hosting Morphik

For those who prefer self-hosting, find detailed instructions [here](https://morphik.ai/docs/getting-started). We provide options for direct installation and Docker installation.

**Important Note:** Due to limited resources, full support for self-hosted deployments is not guaranteed. We offer an installation guide and a [Discord community](https://discord.gg/BwMtv3Zaju) for assistance, but we cannot provide comprehensive support.

## Using Morphik

Once you've signed up, you can immediately start ingesting and searching your data.

### Code Example (Python SDK)

For developers, we offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check). Here's how simple it is to ingest a file:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Searching and querying your data is just as easy:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

The Morphik Console provides a web-based interface for ingesting, searching, and querying data. Upload files, connect to data sources, and interact with your data all in one place.

### Model Context Protocol

Access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome your contributions!  We appreciate:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

Currently, we're focusing on enhancing speed, integrating more tools, and incorporating valuable research papers. Share your ideas in our Discord or on GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.