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

## Morphik: Unlock the Power of Multimodal Data for Your AI Applications

Morphik is an AI-native toolset designed to revolutionize how developers integrate context into their applications by providing end-to-end solutions for storing, representing, and searching unstructured and multimodal data.  Explore the [Morphik Core repository](https://github.com/morphik-org/morphik-core) for the open-source foundation of this powerful tool.

**Important: Migration Required!**  If you installed Morphik before June 22nd, 2025, please run the provided migration script for optimized authentication.

```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

**Key Features:**

*   **Multimodal Search:** Go beyond text-based search and truly understand the visual content of documents. Search images, PDFs, videos, and more with our ColPali technology.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs quickly and easily, with customizable system prompts.
*   **Fast and Scalable Metadata Extraction:**  Efficiently extract metadata, including bounding boxes, labeling, and classification, from your documents.
*   **Integrations:** Seamlessly integrate with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:**  Create persistent KV-caches for faster generation and optimized performance.

### Table of Contents

*   [Why Morphik?](#why)
*   [Getting Started (Recommended)](#getting-started-with-morphik-recommended)
*   [Self-Hosting](#self-hosting-the-open-source-version)
*   [Using Morphik](#using-morphik)
*   [Contributing](#contributing)
*   [License](#license)

### Why Morphik?

Tired of struggling with complex RAG pipelines that fail to understand visually rich documents? Morphik eliminates the need for duct-taped solutions.  We tackle the challenges of unstructured and multimodal data head-on, providing a robust platform that goes beyond traditional methods, ensuring your AI applications are accurate, efficient, and understand the full context of your data.

### Getting Started with Morphik (Recommended)

The easiest way to get started is to sign up for free at [Morphik](https://www.morphik.ai/signup).

### Self-Hosting Morphik

For those who prefer to self-host, detailed instructions are available at [Morphik's Getting Started Guide](https://morphik.ai/docs/getting-started), with options for direct installation and Docker.  Please note that full support is limited.  Join our [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

### Using Morphik

Once signed up, you can begin ingesting and searching your data immediately.

#### Code Example (Python SDK)

Our Python SDK simplifies integration:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

query_result = morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

#### Morphik Console

Utilize the web-based Morphik Console to ingest, search, and query your data.

#### Model Context Protocol

Access Morphik via MCP; instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

### Contributing

We welcome contributions!

*   Report bugs via [GitHub Issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggest features via [GitHub Issues](https://github.com/morphik-org/morphik-core/issues)
*   Submit pull requests

We're currently focused on improving speed, expanding integrations, and leveraging valuable research. Share your thoughts in our Discord or on GitHub.

### License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

### Contributors

Special thanks to our contributors, acknowledged on our [special thanks page](https://morphik.ai/docs/special-thanks).