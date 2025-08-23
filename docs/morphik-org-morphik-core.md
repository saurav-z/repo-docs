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

## Morphik: The AI-Native Platform for Intelligent Document Understanding

Morphik empowers developers to build AI applications that truly understand complex, visually rich documents. **[Explore the Morphik Core repository on GitHub](https://github.com/morphik-org/morphik-core) to get started.**

**Migration Notice:**  If you installed Morphik before June 22nd, 2025, please run the provided migration script to optimize authentication and significantly improve query performance.

```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

### Key Features

*   **Multimodal Search:**  Go beyond text and search images, PDFs, videos, and more with a single endpoint using advanced techniques like ColPali.
*   **Knowledge Graph Generation:**  Build domain-specific knowledge graphs effortlessly with customizable system prompts.
*   **Fast Metadata Extraction:**  Quickly extract essential metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Seamless Integrations:** Connect Morphik to existing tools like Google Suite, Slack, and Confluence for streamlined workflows.
*   **Cache-Augmented Generation:**  Improve generation speed and reduce costs with persistent KV-caches.

### Why Choose Morphik?

Tired of piecing together fragile RAG pipelines that fail with real-world data? Morphik offers a unified platform designed for the complexities of unstructured and multimodal data.  Stop struggling with:

*   **Fragile Pipelines:** Traditional approaches often break down under load.
*   **Limited Understanding:**  Missed information in images, diagrams, and tables.
*   **Performance Bottlenecks:** Exploding infrastructure costs due to redundant processing.

### Get Started

The easiest way to experience Morphik is with our free tier: [Sign up at Morphik](https://www.morphik.ai/signup).

### Table of Contents

*   [Getting Started with Morphik (Recommended)](#getting-started-with-morphik-recommended)
*   [Self-Hosting Morphik](#self-hosting-morphik)
*   [Using Morphik](#using-morphik)
*   [Contributing](#contributing)
*   [License](#license)

### Getting Started with Morphik (Recommended)

Sign up for a free account at [Morphik](https://www.morphik.ai/signup) to start exploring Morphik's capabilities.

### Self-Hosting Morphik

For self-hosting instructions, refer to the dedicated guide: [Self-hosting instructions](https://morphik.ai/docs/getting-started).  Please note that full support for self-hosted deployments is limited.  Join our [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

### Using Morphik

Once you have a Morphik account, you can quickly ingest and search your data.

#### Python SDK Example

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

query_result = morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
print(query_result)
```

#### Morphik Console

Use the web-based Morphik Console for easy data ingestion, search, and querying: upload files, connect data sources, and interact with your data in one place.

#### Model Context Protocol (MCP)

Access Morphik via MCP: [MCP instructions](https://morphik.ai/docs/using-morphik/mcp).

### Contributing

We welcome contributions!  Help us by:

*   Reporting bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggesting feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submitting pull requests

Our current focus is on improving speed, expanding integrations, and staying at the forefront of research.  Share your thoughts on Discord or GitHub!

### License

Morphik Core is source-available under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full license text for details.

### Contributors

Special thanks to our contributors!  See the list: [Contributors page](https://morphik.ai/docs/special-thanks)