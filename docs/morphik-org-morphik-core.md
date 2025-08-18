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

[View the Morphik Core Repository on GitHub](https://github.com/morphik-org/morphik-core)

<!-- add a roadmap! - <a href="https://morphik.ai/roadmap">Roadmap</a> - -->
<!-- Add a changelog! - <a href="https://morphik.ai/changelog">Changelog</a> -->

<p align="center">
  <a href="https://morphik.ai/docs">Docs</a> - <a href="https://discord.gg/BwMtv3Zaju">Community</a> - <a href="https://morphik.ai/docs/blogs/gpt-vs-morphik-multimodal">Why Morphik?</a> - <a href="https://github.com/morphik-org/morphik-core/issues/new?assignees=&labels=bug&template=bug_report.md">Bug reports</a>
</p>

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Morphik: Revolutionizing AI Applications with Multimodal Data Understanding

Morphik empowers developers to build smarter AI applications by providing the tools to ingest, search, transform, and manage unstructured and multimodal documents with ease.

**Key Features:**

*   **Multimodal Search:** Unlock a new level of data understanding with search that *understands* the visual content of documents, including images, PDFs, and videos, using techniques like ColPali.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs effortlessly with a single line of code, using battle-tested system prompts or your custom ones.
*   **Fast and Scalable Metadata Extraction:** Efficiently extract metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Seamless Integrations:** Integrate with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Speed up generation with persistent KV-caches of your documents.
*   **Free Tier:** Get started with Morphik's generous free tier and transparent, compute-usage based pricing.

## Table of Contents

*   [Getting Started with Morphik](#getting-started-with-morphik-recommended)
*   [Self-hosting Morphik](#self-hosting-the-open-source-version)
*   [Using Morphik](#using-morphik)
*   [Contributing](#contributing)
*   [License](#license)

## Getting Started with Morphik (Recommended)

The quickest and simplest way to get started is by signing up for a free account at [Morphik](https://www.morphik.ai/signup).

## Self-hosting Morphik

For those who prefer self-hosting, instructions are available [here](https://morphik.ai/docs/getting-started), with options for direct installation and Docker.

**Important:**  Due to resource limitations, we can't offer full support for self-hosted deployments. Refer to our installation guide and [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

## Using Morphik

Once signed up, you can immediately begin ingesting and searching your data.

### Code Example (Python SDK)

For programmers, we offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check). Ingesting a file is straightforward:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Querying your data is also easy:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

The Morphik Console, a web-based interface, allows you to ingest, search, and query your data. You can upload files, connect to various data sources, and interact with your data.

### Model Context Protocol

Instructions for accessing Morphik via MCP are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions!  Please submit:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

We're currently focusing on improving speed, expanding integrations, and identifying valuable research papers.

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.