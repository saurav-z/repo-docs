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

## Morphik: Unlock the Power of Multimodal Data for Your AI Applications

**[Morphik Core](https://github.com/morphik-org/morphik-core) is an AI-native toolkit designed to revolutionize how you integrate complex data into your AI applications.**

### Key Features:

*   **Multimodal Search:**  Go beyond text-based search and understand the visual content of documents including images, PDFs, and videos, using cutting-edge techniques like ColPali.
*   **Knowledge Graph Creation:** Build domain-specific knowledge graphs effortlessly with a single line of code, leveraging our pre-built system prompts or creating your own.
*   **Fast Metadata Extraction:** Quickly extract valuable metadata such as bounding boxes, labels, and classifications from your documents.
*   **Seamless Integrations:** Connect with your existing tools and workflows, including Google Suite, Slack, Confluence, and more.
*   **Cache-Augmented Generation:**  Boost performance and reduce costs with persistent KV-caches of your documents.

### Why Morphik?

Tired of brittle RAG pipelines and AI applications that misunderstand visually rich documents? Morphik solves the limitations of traditional approaches by offering a unified platform for:

*   **Ingestion & Storage:**  Handle unstructured and multimodal documents with ease.
*   **Deep & Shallow Search:**  Find the exact information you need.
*   **Transformation & Management:**  Transform, manage and use your data to accelerate your AI development.

### Get Started with Morphik

The easiest way to experience Morphik is by signing up for a free account at [Morphik](https://www.morphik.ai/signup).  We offer a generous free tier and transparent, compute-usage based pricing.

### Self-Hosting Morphik (Open Source)

If you prefer to self-host, the open-source version of Morphik is available. Find detailed instructions at [Morphik Documentation](https://morphik.ai/docs/getting-started).

**Important:** Self-hosting support is limited. Please refer to the installation guide and our [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

### Using Morphik

Once you've signed up, start ingesting and searching your data immediately.

#### Code Example (Python SDK)

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

#### Morphik Console

Use the web-based interface to ingest, search, and query your data. Upload files, connect to data sources, and interact with your data within the console.

#### Model Context Protocol (MCP)

Access Morphik through MCP; instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

### Contributing

We welcome contributions!  Help us by:

*   Reporting bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Suggesting features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Submitting pull requests.

Currently, we're focused on: improving speed, integrating more tools, and incorporating valuable research. Share your thoughts on Discord or GitHub!

### License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

### Contributors

View our [special thanks page](https://morphik.ai/docs/special-thanks) to see our contributors.