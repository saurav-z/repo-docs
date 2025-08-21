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

Morphik simplifies integrating context into your AI applications, offering a complete toolkit to store, represent, and search unstructured and multimodal data. [Explore the source code on GitHub](https://github.com/morphik-org/morphik-core).

**Migration Required for Existing Installations:** If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
```bash
python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
```

**Key Features:**

*   **Multimodal Search:** Understand the visual content of documents with search that understands images, PDFs, videos, and more.
*   **Knowledge Graphs:** Build domain-specific knowledge graphs quickly and easily, even in a single line of code.
*   **Fast Metadata Extraction:** Efficiently extract metadata like bounding boxes, labels, and classifications from your documents.
*   **Integrations:** Seamlessly connect with popular tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented Generation:** Speed up generation and reduce costs with persistent KV-caches for your documents.

**Why Morphik?**

Stop cobbling together disparate tools for your AI applications. Morphik provides a comprehensive solution, eliminating the need for fragile, complex pipelines and addressing the limitations of traditional RAG approaches. Avoid errors, unlock information hidden in visual data, and dramatically improve performance.

**Getting Started with Morphik (Recommended)**

The easiest way to get started is by signing up for free at [Morphik](https://www.morphik.ai/signup). We offer a generous free tier and compute-usage based pricing.

**Self-hosting Morphik**

For self-hosting instructions, please refer to the documentation [here](https://morphik.ai/docs/getting-started). Note: We cannot guarantee full support for self-hosted deployments.

**Using Morphik**

Once you've signed up or self-hosted Morphik, you can ingest and search your data immediately.

### Code (Example: Python SDK)

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

### Querying your data

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Interact with Morphik via the web-based Morphik Console. Upload files, connect data sources, and chat with your data.

### Model Context Protocol

Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

**Contributing**

We welcome contributions!  Report bugs, suggest features, or submit pull requests.

*   Bug reports: [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests: [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

**License**

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
  Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

**Contributors**

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.