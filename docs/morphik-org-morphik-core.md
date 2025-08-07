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

## Morphik: Unlock the Power of Multimodal Data for AI Applications

Morphik empowers developers to build AI applications that truly understand and interact with complex, visually rich documents and multimodal data, offering a comprehensive suite of tools for ingestion, search, transformation, and management.  Explore the source code on [GitHub](https://github.com/morphik-org/morphik-core).

**Migration Note:**  If you installed Morphik before June 22nd, 2025, please run the migration script to optimize authentication performance. See the original README for migration instructions.

**Key Features:**

*   **Multimodal Search:** Search across images, PDFs, videos, and more with a single endpoint, understanding the visual content of your documents.
*   **Knowledge Graphs:** Build custom knowledge graphs for domain-specific use cases with ease.
*   **Fast and Scalable Metadata Extraction:** Extract valuable metadata, including bounding boxes, labels, and classifications.
*   **Seamless Integrations:** Connect Morphik with your existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Accelerate generation with persistent KV-caches of your documents.

## Table of Contents

*   [Getting Started with Morphik (Recommended)](#getting-started-with-morphik-recommended)
*   [Self-hosting Morphik](#self-hosting-morphik)
*   [Using Morphik](#using-morphik)
*   [Contributing](#contributing)
*   [License](#license)

## Getting Started with Morphik (Recommended)

The easiest way to get started is by signing up for a free account at [Morphik](https://www.morphik.ai/signup).  Enjoy our generous free tier and transparent, usage-based pricing.

## Self-hosting Morphik

For those who prefer self-hosting, find detailed instructions [here](https://morphik.ai/docs/getting-started). We offer both direct installation and Docker options.

**Important:**  Due to limited resources, we offer best-effort support for self-hosted deployments.  Refer to the installation guide and the [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

## Using Morphik

Once you have access to Morphik, start ingesting and searching your data immediately.

### Code (Example: Python SDK)

The Python SDK provides a simple interface for developers.

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

result = morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
print(result)
```

### Morphik Console

The web-based Morphik Console allows you to ingest, search, query and manage your data within a single place.

### Model Context Protocol

Access Morphik through MCP; find instructions [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome your contributions!

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Suggest new features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Submit pull requests.

Currently we're focused on:
* Speed Improvements
* New Tool Integrations
* Refining Research

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.