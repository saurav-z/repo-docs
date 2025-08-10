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

## Morphik: Revolutionizing AI with Multimodal Data Understanding

Morphik empowers developers to build AI applications that truly understand complex documents and multimodal data.  Explore the [Morphik Core GitHub repository](https://github.com/morphik-org/morphik-core) to learn more.

### Key Features

*   **Multimodal Search:** Go beyond text; search images, PDFs, videos, and more with ColPali technology.
*   **Knowledge Graphs:** Quickly build domain-specific knowledge graphs with a single line of code.
*   **Fast Metadata Extraction:** Extract valuable information from documents, including bounding boxes and classifications.
*   **Seamless Integrations:** Connect to existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Speed up generation with persistent KV-caches of your documents.

### Why Morphik?

Traditional RAG approaches often struggle with real-world data, especially visually rich documents. Morphik solves this by:

*   **Addressing RAG Limitations:** Overcoming the limitations of traditional Retrieval-Augmented Generation (RAG) systems.
*   **Understanding Visuals:** Enabling AI applications to truly understand images, charts, diagrams, and other visual elements.
*   **Improving Accuracy:** Ensuring AI applications provide accurate answers by comprehending all data types.
*   **Boosting Performance:** Reducing infrastructure costs and speeding up query processing with efficient data handling.

### Getting Started

The easiest way to start with Morphik is to sign up for the [free tier](https://www.morphik.ai/signup) on the Morphik website.

### Migration Notice

If you installed Morphik before June 22nd, 2025, please run the following migration script to optimize performance:

```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

###  Self-Hosting

For self-hosting instructions, refer to the [Morphik documentation](https://morphik.ai/docs/getting-started). Note that self-hosted deployments have limited support. Join our [Discord community](https://discord.gg/BwMtv3Zaju) for help!

###  Using Morphik

Once you're set up, start ingesting and querying your data immediately.

####  Code Example (Python SDK)

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

###  Morphik Console

Use the web-based console to ingest, search, and query your data.

###  Model Context Protocol (MCP)

Access Morphik via the Model Context Protocol; instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

### Contributing

We welcome contributions!

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Suggest features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Submit pull requests.

### License

Morphik Core is source-available under the [Business Source License 1.1](./LICENSE).

-   **Personal / Indie use**: free.
-   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
-   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

### Contributors

See our [special thanks page](https://morphik.ai/docs/special-thanks) for a list of contributors.