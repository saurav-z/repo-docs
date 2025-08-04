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

## Morphik: Revolutionize AI Applications with Multimodal Data Understanding

Morphik is a powerful AI-native toolset that allows developers to seamlessly integrate complex context and multimodal data into their AI applications.  Explore the open-source core on [GitHub](https://github.com/morphik-org/morphik-core).

**Migration Notice:** *If you installed Morphik before June 22nd, 2025, run the migration script for faster query performance:*
```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

### Key Features

*   **Multimodal Search:** Go beyond text and search over images, PDFs, videos, and more with our ColPali technology, understanding the visual content within documents.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs effortlessly, with pre-built system prompts or your own custom configurations.
*   **Fast Metadata Extraction:** Quickly extract metadata from your documents, including bounding boxes, labeling, and classifications.
*   **Seamless Integrations:** Connect with popular tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Boost generation speeds by creating persistent KV-caches of your documents.

### Why Morphik?

Morphik solves the complexities of building AI applications that interact with diverse data types. Traditional RAG approaches often fail in production due to fragile pipelines and the inability to understand visually rich documents. Morphik tackles these challenges head-on, providing a robust solution for handling complex, unstructured, and multimodal data.  Sign up for the [free tier](https://www.morphik.ai/signup) to get started!

### Getting Started

#### Recommended: Using the Morphik Cloud Platform
The easiest way to get started with Morphik is by signing up for a free account at [Morphik](https://www.morphik.ai/signup). This gives you access to our platform with a generous free tier and transparent, compute-usage based pricing.

#### Self-Hosting (Open Source)

For those who prefer self-hosting, the open-source version of Morphik provides the core functionality.

1.  **Installation:** Follow the detailed instructions in our [documentation](https://morphik.ai/docs/getting-started).
2.  **Support:** Community support is available via our [Discord community](https://discord.gg/BwMtv3Zaju), though full support for self-hosted deployments is limited due to resource constraints.

#### Using Morphik

After signing up, you can immediately begin ingesting and searching your data.

**1. Code Example (Python SDK)**

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

**2. Morphik Console**

Interact with Morphik via a web-based interface to ingest, search, and query your data.

**3. Model Context Protocol (MCP)**

Access Morphik through the MCP.  Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

### Contributing

We welcome contributions! Help us improve Morphik by:

*   Reporting bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Suggesting new features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Submitting Pull Requests.

Currently, we're focusing on speed improvements, tool integrations, and exploring impactful research papers.  Share your thoughts in our Discord or on GitHub!

### License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

### Contributors

A special thanks to all our contributors!  Visit our [special thanks page](https://morphik.ai/docs/special-thanks) for recognition.