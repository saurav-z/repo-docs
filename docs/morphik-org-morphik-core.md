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

## Morphik: Unlock the Power of Multimodal Data for AI Applications

Morphik is an AI-native toolset designed to empower developers to seamlessly integrate complex and nuanced context into their AI applications by providing end-to-end solutions for storing, representing, and searching unstructured and multimodal data. This repository contains the open-source core of Morphik. Find the original source code on [GitHub](https://github.com/morphik-org/morphik-core).

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features

*   **Multimodal Search:** Go beyond text and search over images, PDFs, videos, and more with a single endpoint, using advanced techniques like ColPali to understand visual content.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs with ease, using pre-built system prompts or your own custom configurations.
*   **Fast Metadata Extraction:** Quickly extract valuable metadata from documents, including bounding boxes, labels, and classifications.
*   **Integrations:** Integrate with your existing tools and workflows.  Currently, we provide integrations with Google Suite, Slack, Confluence, and more.
*   **Cache-Augmented-Generation:** Utilize persistent KV-caches to speed up document generation.

## Why Morphik?

Traditional approaches to building AI applications for data often fall short, especially when dealing with complex, visually rich documents. Existing solutions struggle to extract meaning from charts, diagrams, and tables, leading to inaccurate results and inefficient processing. Morphik solves these issues by providing a cohesive and powerful platform for:

*   **Improved Accuracy:** Truly understand and analyze visual data, ensuring your AI applications provide correct and reliable answers.
*   **Reduced Costs:** Optimize performance and minimize infrastructure expenses with efficient data processing and retrieval.
*   **Simplified Development:** Eliminate the need to stitch together disparate tools, streamline your workflow, and accelerate your development cycle.

## Getting Started

The easiest way to get started with Morphik is through our free tier!  Sign up at [Morphik](https://www.morphik.ai/signup).

### Table of Contents
- [Getting Started with Morphik (Recommended)](#getting-started-with-morphik-recommended)
- [Self-hosting Morphik](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [License](#license)

## Getting Started with Morphik (Recommended)

Sign up for a free account at [Morphik](https://www.morphik.ai/signup) to begin using Morphik.

## Self-hosting Morphik

If you'd like to self-host Morphik, see the dedicated instruction [here](https://morphik.ai/docs/getting-started). We offer options for direct installation and installation via docker.

**Important**: Due to limited resources, we cannot provide full support for self-hosted deployments. We have an installation guide, and a [Discord community](https://discord.gg/BwMtv3Zaju) to help, but we can't guarantee full support.

## Using Morphik

### Code (Example: Python SDK)

For programmers, we offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check).

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Interact with Morphik via the Morphik Console, a web-based interface that allows you to ingest, search, and query data.

### Model Context Protocol

Access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions!

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggest features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submit pull requests

We are currently focused on:

*   Improving speed
*   Integrating more tools
*   Identifying valuable research papers

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

-   **Personal / Indie use**: free.
-   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
-   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.