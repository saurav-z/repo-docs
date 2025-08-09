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

Morphik is a powerful AI-native toolset designed to revolutionize how developers integrate context into their AI applications by enabling advanced search and data management across various data formats.  Explore the [Morphik Core](https://github.com/morphik-org/morphik-core) repository for more information and contributions.

**Key Features:**

*   **Multimodal Search:**  Go beyond text-based search with ColPali and other techniques to understand the *visual* content of documents including images, PDFs, videos, and more.
*   **Knowledge Graphs:** Easily build domain-specific knowledge graphs directly within your applications using built-in or custom system prompts.
*   **Fast & Scalable Metadata Extraction:** Quickly extract valuable metadata like bounding boxes, labels, and classifications from documents.
*   **Seamless Integrations:**  Connect Morphik with your existing workflows using integrations with tools like Google Suite, Slack, and Confluence.
*   **Cache-Augmented Generation:**  Optimize performance and reduce costs using persistent KV-caches for faster generation.
*   **Free Tier:**  Get started with Morphik today with our generous free tier! [Sign up now](https://www.morphik.ai/signup).

## Table of Contents

*   [Getting Started with Morphik](#getting-started-with-morphik-recommended)
*   [Self-hosting Morphik](#self-hosting-morphik)
*   [Using Morphik](#using-morphik)
*   [Contributing](#contributing)
*   [License](#license)

## Getting Started with Morphik (Recommended)

The quickest way to experience Morphik is by signing up for our free tier at [Morphik](https://www.morphik.ai/signup). Explore the power of advanced multimodal search today.

## Self-hosting Morphik

For those who prefer self-hosting, detailed instructions are available [here](https://morphik.ai/docs/getting-started), including direct installation and Docker options.

**Important Note:** Due to limited resources, full support for self-hosted deployments is not guaranteed.  Refer to our installation guide and [Discord community](https://discord.gg/BwMtv3Zaju) for assistance.

## Using Morphik

Once you've signed up, get started with ingesting and searching your data right away.

### Code (Example: Python SDK)

We offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check). Ingesting a file is as simple as:

```python
from morphik import Morphik

morphik = Morphik("<your-morphik-uri>")
morphik.ingest_file("path/to/your/super/complex/file.pdf")
```

Querying is just as simple:

```python
morphik.query("What's the height of screw 14-A in the chair assembly instructions?")
```

### Morphik Console

Interact with Morphik via the web-based Morphik Console to upload files, search your data, and chat with your data.

### Model Context Protocol

Access Morphik via MCP; instructions are available [here](https://morphik.ai/docs/using-morphik/mcp).

## Contributing

We welcome contributions!  Consider:

*   Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
*   Pull requests

Currently, we're focused on improving speed, tool integrations, and research. Share your ideas on Discord or GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

*   **Personal / Indie use**: free.
*   **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
    Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
*   **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full license text for details.

## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.