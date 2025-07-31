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

## Morphik: Powering AI Applications with Deep Understanding of Multimodal Data

Tired of RAG systems that fail on complex documents?  **Morphik is a powerful, AI-native toolkit designed to help developers build intelligent applications that truly understand and interact with visually rich, multimodal data.**  Learn more at the [Morphik Core GitHub repository](https://github.com/morphik-org/morphik-core).

**Important Note for Existing Installations:**  If you installed Morphik before June 22nd, 2025, please run the migration script for optimal performance.  Run the following command before launching Morphik:

```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

## Key Features

Morphik offers a comprehensive suite of tools to ingest, search, transform, and manage your unstructured and multimodal documents:

*   **Multimodal Search:** Search across images, PDFs, videos, and more, understanding the visual content with technologies like ColPali.
*   **Knowledge Graphs:** Easily build knowledge graphs for your domain-specific use cases.
*   **Fast Metadata Extraction:** Quickly extract crucial metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Seamless Integrations:** Connect with existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented-Generation:** Improve performance and reduce costs with persistent KV-caches of your documents.
*   **Free Tier:** Get started today with Morphik's generous free tier at [Morphik](https://www.morphik.ai/signup).

## Why Morphik?

Traditional RAG systems struggle with the complexity of real-world documents, often failing to understand the nuances of visual data. Morphik addresses these shortcomings by:

*   **Understanding Visuals:**  Morphik's multimodal capabilities ensure that charts, diagrams, and other visual elements are understood, not just ignored.
*   **Scalability and Performance:**  Morphik is designed to handle large datasets and complex queries efficiently, reducing infrastructure costs.
*   **Simplified Integration:**  Morphik provides a streamlined solution, eliminating the need to duct-tape together multiple tools.

## Getting Started

### Recommended: Sign Up for Morphik (Free Tier Available)

The easiest way to get started is by signing up for a free account at [Morphik](https://www.morphik.ai/signup).

### Self-hosting Morphik

For self-hosting instructions, refer to the [dedicated guide](https://morphik.ai/docs/getting-started).

**Important:** Due to limited resources, full support is not guaranteed for self-hosted deployments. However, you can get help from our installation guide and our [Discord community](https://discord.gg/BwMtv3Zaju).

## Using Morphik

Once you're signed up, you can start ingesting and querying your data.

### Code (Example: Python SDK)

Morphik offers a Python SDK and a REST API for programmatic access:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

The web-based console allows you to ingest, search, and query your data via an intuitive interface.

### Model Context Protocol

Access Morphik through MCP; instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions! Please feel free to:

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggest features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submit pull requests

Currently, we're focusing on speed, integrations, and research-driven improvements. Share your thoughts on Discord or GitHub!

## License

Morphik Core is source-available under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.  Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) to see the wonderful people who have contributed.