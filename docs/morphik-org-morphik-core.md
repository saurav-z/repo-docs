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

## Morphik: Unlock the Power of Your Multimodal Data with AI

Morphik provides a powerful, end-to-end platform for integrating complex data, including visually rich documents, into your AI applications.  Explore the source code on [GitHub](https://github.com/morphik-org/morphik-core).

### Key Features:

*   **Multimodal Search:**  Go beyond text-based search; understand the visual content of your documents. Search over images, PDFs, videos, and more with a single endpoint using techniques like ColPali.
*   **Knowledge Graph Creation:** Easily build domain-specific knowledge graphs. Utilize our pre-built system prompts or integrate your own.
*   **Fast & Scalable Metadata Extraction:** Efficiently extract metadata, including bounding boxes, labels, and classifications, from your documents.
*   **Seamless Integrations:** Connect Morphik with your existing tools and workflows, including Google Suite, Slack, and Confluence.
*   **Cache-Augmented Generation:** Accelerate your AI applications with persistent KV-caches of your documents.
*   **Free Tier:** Get started today with Morphik's generous free tier.

### Table of Contents
- [Getting Started with Morphik (Recommended)](#getting-started-with-morphik-recommended)
- [Self-hosting Morphik](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [License](#license)

## Getting Started with Morphik (Recommended)

The fastest and easiest way to get started with Morphik is by signing up for free at [Morphik](https://www.morphik.ai/signup). We have a generous free tier and transparent, compute-usage based pricing if you're looking to ingest a lot of data.

## Self-hosting Morphik
If you'd like to self-host Morphik, you can find the dedicated instruction [here](https://morphik.ai/docs/getting-started). We offer options for direct installation and installation via docker.

**Important**: Due to limited resources, we cannot provide full support for self-hosted deployments. We have an installation guide, and a [Discord community](https://discord.gg/BwMtv3Zaju) to help, but we can't guarantee full support.

## Using Morphik

Once you've signed up for Morphik, you can get started with ingesting and searching your data right away.

### Code (Example: Python SDK)

For developers, we offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check). Ingesting a file is as simple as:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Similarly, searching and querying your data is easy too:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

You can also interact with Morphik via the Morphik Console. This is a web-based interface that allows you to ingest, search, and query your data. You can upload files, connect to different data sources, and chat with your data all within the same place.

### Model Context Protocol

Finally, you can also access Morphik via MCP. Instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions! Help us improve Morphik by:

*   Reporting bugs via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Suggesting new features via [GitHub issues](https://github.com/morphik-org/morphik-core/issues).
*   Submitting pull requests.

We're currently focused on performance improvements, tool integrations, and leveraging relevant research.

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.