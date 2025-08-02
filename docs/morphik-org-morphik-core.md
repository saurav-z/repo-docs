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

> **Migration Required for Existing Installations**: If you installed Morphik before June 22nd, 2025, we've optimized our authentication system for 70-80% faster query performance. Please run the migration script before launching Morphik:
> ```bash
> python scripts/migrate_auth_columns_complete.py --postgres-uri "postgresql+asyncpg://user:pass@host:port/db"
> ```

## Morphik is a AI-native toolset for visually rich documents and multimodal data

We are building the best way for developers to integrate context (however complex and nuanced) into their AI applications. We offer a treasure chest of tools to store, represent, and search (shallow, and deep) unstructured data. End-to-End.

## Why?

Building AI applications that interact with data shouldn't require duct-taping together a dozen different tools just to get relevant results to your LLM.

Traditional RAG approaches that work in proof-of-concepts often fail spectacularly in production. Cobbling together separate systems for text extraction, OCR, embeddings, vector databases, and retrieval creates fragile pipelines that break under real-world load. Each component brings its own APIs, configurations, and failure modes - what starts as a simple demo becomes an unmaintainable mess at scale.

Even worse, these pipelines fundamentally fail at understanding visually rich documents. Charts become meaningless text fragments. Critical diagrams lose their spatial relationships. Tables get mangled into unreadable strings. Technical specifications with mixed text and visuals? Forget about accuracy.

The result is AI applications that confidently return wrong answers because they never truly understood the documents. They miss crucial information embedded in images, misinterpret technical diagrams, and treat visual data as an afterthought. And performance? Watch your infrastructure costs explode as your LLM re-processes the same 500-page manual for every single query.

## What?
[Morphik](https://morphik.ai) provides developers the tools to ingest, search (deep and shallow), transform, and manage unstructured and multimodal documents. Some of our features include:

- [Multimodal Search](https://morphik.ai/docs/concepts/colpali): We employ techniques such as ColPali to build search that actually *understands* the visual content of documents you provide. Search over images, PDFs, videos, and more with a single endpoint.
- [Knowledge Graphs](https://morphik.ai/docs/concepts/knowledge-graphs): Build knowledge graphs for domain-specific use cases in a single line of code. Use our battle-tested system prompts, or use your own.
- [Fast and Scalable Metadata Extraction](https://morphik.ai/docs/concepts/rules-processing): Extract metadata from documents - including bounding boxes, labeling, classification, and more.
- [Integrations](https://morphik.ai/docs/integrations): Integrate with existing tools and workflows. Including (but not limited to) Google Suite, Slack, and Confluence.
- [Cache-Augmented-Generation](https://morphik.ai/docs/python-sdk/create_cache): Create persistent KV-caches of your documents to speed up generation.

The best part? Morphik has a [free tier](https://www.morphik.ai/pricing)! Get started by signing up at [Morphik](https://www.morphik.ai/signup).

## Table of Contents
- [Getting Started with Morphik](#getting-started-with-morphik-recommended)
- [Self-hosting Morphik](#self-hosting-the-open-source-version)
- [Using Morphik](#using-morphik)
- [Contributing](#contributing)
- [Open source vs paid](#License)

## Getting Started with Morphik (Recommended)

The fastest and easiest way to get started with Morphik is by signing up for free at [Morphik](https://www.morphik.ai/signup). We have a generous free tier and transparent, compute-usage based pricing if you're looking to ingest a lot of data.

## Self-hosting Morphik
If you'd like to self-host Morphik, you can find the dedicated instruction [here](https://morphik.ai/docs/getting-started). We offer options for direct installation and installation via docker.

**Important**: Due to limited resources, we cannot provide full support for self-hosted deployments. We have an installation guide, and a [Discord community](https://discord.gg/BwMtv3Zaju) to help, but we can't guarantee full support.

## Using Morphik

Once you've signed up for Morphik, you can get started with ingesting and searching your data right away.


### Code (Example: Python SDK)
For programmers, we offer a [Python SDK](https://morphik.ai/docs/python-sdk/morphik) and a [REST API](https://morphik.ai/docs/api-reference/health-check). Ingesting a file is as simple as:

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
You're welcome to contribute to the project! We love:
- Bug reports via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
- Feature requests via [GitHub issues](https://github.com/morphik-org/morphik-core/issues)
- Pull requests

Currently, we're focused on improving speed, integrating with more tools, and finding the research papers that provide the most value to our users. If you have thoughts, let us know in the discord or in GitHub!

## License

Morphik Core is **source-available** under the [Business Source License 1.1](./LICENSE).

- **Personal / Indie use**: free.
- **Commercial production use**: free if your Morphik deployment generates < $2 000/month in gross revenue.
  Otherwise purchase a commercial key at <https://morphik.ai/pricing>.
- **Future open source**: each code version automatically re-licenses to Apache 2.0 exactly four years after its first release.

See the full licence text for details.


## Contributors

Visit our [special thanks page](https://morphik.ai/docs/special-thanks) dedicated to our contributors.
