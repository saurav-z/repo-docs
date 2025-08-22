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

## Morphik: Unlock the Power of Your Data with AI-Powered Document Understanding

Morphik provides a powerful AI-native toolset, enabling developers to seamlessly integrate complex context and unlock deep insights from visually rich documents and multimodal data.  Explore the open-source [Morphik Core](https://github.com/morphik-org/morphik-core) and learn how you can transform your AI applications.

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Key Features of Morphik

*   **Multimodal Search**:  Go beyond text with search that *understands* visual content, enabling you to search images, PDFs, videos, and more with a single query. ([More Info](https://morphik.ai/docs/concepts/colpali))
*   **Knowledge Graphs**: Build domain-specific knowledge graphs with ease using our battle-tested system prompts or customize with your own. ([More Info](https://morphik.ai/docs/concepts/knowledge-graphs))
*   **Fast Metadata Extraction**: Efficiently extract key metadata from your documents, including bounding boxes, classifications, and more. ([More Info](https://morphik.ai/docs/concepts/rules-processing))
*   **Seamless Integrations**: Connect Morphik with your existing tools and workflows, including Google Suite, Slack, Confluence, and many others. ([More Info](https://morphik.ai/docs/integrations))
*   **Cache-Augmented Generation**: Accelerate generation and reduce costs with persistent KV-caches of your documents. ([More Info](https://morphik.ai/docs/python-sdk/create_cache))

## Why Choose Morphik?

Tired of struggling to make sense of your unstructured data?  Morphik solves the challenges of building production-ready AI applications that interact with complex documents. Unlike traditional RAG approaches that often fail with visually rich content, Morphik allows you to understand images, diagrams, and tables.

## Get Started

The easiest way to get started with Morphik is by signing up for the [free tier](https://www.morphik.ai/signup).

## Table of Contents
- [Getting Started with Morphik (Recommended)](#getting-started-with-morphik-recommended)
- [Self-hosting Morphik](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [License](#license)

## Getting Started with Morphik (Recommended)

Sign up for a free account at [Morphik](https://www.morphik.ai/signup) to quickly begin using Morphik.

## Self-hosting Morphik

For self-hosting instructions, please refer to the dedicated guide [here](https://morphik.ai/docs/getting-started).  We offer options for direct installation and Docker-based deployments.

**Important**: We can't provide full support for self-hosted deployments due to limited resources.

## Using Morphik

Once you've signed up for Morphik, you can start ingesting and searching your data right away.

### Code (Example: Python SDK)

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")

morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Use the web-based interface for ingesting, searching, and querying data.

### Model Context Protocol

Access Morphik via MCP, instructions are [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome your contributions!

*   Report bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Suggest features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Submit pull requests

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) to see our contributors.