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

## Morphik: Unlock the Power of Your Data with AI-Native Search and Knowledge Management

Tired of AI applications that misunderstand your documents? **Morphik** is the AI-native toolset designed to help developers effortlessly integrate context and unlock the full potential of unstructured and multimodal data. Explore the [Morphik Core GitHub repository](https://github.com/morphik-org/morphik-core) for open-source components.

**Key Features:**

*   **Multimodal Search:** Understand and search visual content within your documents, including images, PDFs, and videos, all through a single endpoint. Learn more about [ColPali](https://morphik.ai/docs/concepts/colpali).
*   **Knowledge Graphs:** Build domain-specific knowledge graphs in a single line of code, leveraging battle-tested system prompts or your own custom configurations.
*   **Fast and Scalable Metadata Extraction:** Efficiently extract metadata from documents, including bounding boxes, labeling, and classification.
*   **Seamless Integrations:** Connect with your existing tools and workflows, including (but not limited to) Google Suite, Slack, and Confluence. See our [Integrations](https://morphik.ai/docs/integrations) documentation.
*   **Cache-Augmented-Generation:** Create persistent KV-caches of your documents to dramatically speed up generation.
*   **[Free Tier Available](https://www.morphik.ai/pricing):** Get started without any commitment!

## Table of Contents

*   [Getting Started with Morphik (Recommended)](#getting-started-with-morphik-recommended)
*   [Self-hosting Morphik](#self-hosting-morphik)
*   [Using Morphik](#using-morphik)
*   [Contributing](#contributing)
*   [License](#license)

## Getting Started with Morphik (Recommended)

The fastest and easiest way to experience Morphik is by signing up for a free account at [Morphik](https://www.morphik.ai/signup). Our generous free tier and transparent, compute-usage-based pricing make it easy to get started.

## Self-hosting Morphik

For those who prefer self-hosting, we provide installation instructions [here](https://morphik.ai/docs/getting-started), including options for direct installation and Docker.

**Important:** Due to limited resources, we can only provide basic support for self-hosted deployments. A [Discord community](https://discord.gg/BwMtv3Zaju) is available to help.

## Using Morphik

Once you've signed up or set up your self-hosted instance, you can immediately begin ingesting and querying your data.

### Code (Example: Python SDK)

Developers can use our [Python SDK](https://morphik.ai/docs/python-sdk/morphik) or [REST API](https://morphik.ai/docs/api-reference/health-check). Ingesting a file is as simple as:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Searching and querying is just as straightforward:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Use the web-based Morphik Console to ingest, search, and query your data. Upload files, connect to data sources, and interact with your data all in one place.

### Model Context Protocol

Access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions! Help us by:

*   Reporting bugs through [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggesting new features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submitting pull requests

We are currently focused on improving performance, expanding integrations, and incorporating cutting-edge research. Share your thoughts on Discord or GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
  Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Special thanks to our contributors. Visit our [special thanks page](https://morphik.ai/docs/special-thanks).

***
**Migration Notice:** If you installed Morphik before June 22nd, 2025, improve query performance by running this migration script:

```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"