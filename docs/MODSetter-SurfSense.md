<!-- Improved README - SurfSense -->

# SurfSense: Your Customizable AI Research Agent 🧠

SurfSense empowers you to conduct advanced research by connecting your personal knowledge base with external sources. This open-source AI research agent is highly customizable and designed to revolutionize how you gather information.  [View the original repository on GitHub](https://github.com/MODSetter/SurfSense)

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord)](https://discord.gg/ejRNvftDp9)

---

**Key Features:**

*   💡 **Private, Customizable AI Research Assistant:** Build your own private NotebookLM and Perplexity, integrated with your data and external sources.
*   📁 **Extensive File Format Support:** Upload and index content from various file formats, supporting over 50 file extensions.
*   🔍 **Powerful Search Capabilities:** Quickly find information within your saved content using semantic and full-text search.
*   💬 **Conversational Research:** Interact with your data in natural language and receive cited answers.
*   📄 **Cited Answers:** Get trustworthy answers with source citations, mirroring the functionality of tools like Perplexity.
*   🔔 **Privacy & Local LLM Support:** Seamlessly integrates with local LLMs using Ollama.
*   🏠 **Self-Hosted:** Open-source and easy to deploy locally, giving you complete control over your data.
*   🎙️ **Advanced Podcast Generation:** Create engaging audio content from your chats or research with lightning-fast podcast creation.
    *   Blazingly fast podcast generation agent (creates a 3-minute podcast in under 20 seconds).
    *   Convert your chat conversations into engaging audio content
    *   Support for local TTS providers (Kokoro TTS)
    *   Support for multiple TTS providers (OpenAI, Azure, Google Vertex AI)
*   📊 **Advanced RAG (Retrieval-Augmented Generation) Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecone, Cohere, Flashrank, etc.)
    *   Utilizes Hierarchical Indices (2 tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **External Source Integrations:**
    *   Search Engines: Tavily, LinkUp.
    *   Collaboration Tools: Slack, Linear, Jira, ClickUp, Confluence, Notion, YouTube.
    *   Code Repositories: GitHub, Discord.
    *   More integrations are continually being added!
*   🔖 **Browser Extension:** Save webpages directly to your knowledge base, even behind authentication.

---

## Supported File Extensions

SurfSense's flexibility allows you to upload and search a wide range of file types.  File format support depends on your ETL service configuration (LlamaCloud, Unstructured, and Docling).

**[Detailed File Extension Lists in Original README]**

---

## Installation

SurfSense offers two installation methods:

1.  **Docker Installation (Recommended):** Easiest setup with all dependencies containerized. Includes pgAdmin for database management, supports environment variable customization, and offers flexible deployment options. [See Docker Setup Guide](https://www.surfsense.net/docs/docker-installation)
2.  **Manual Installation:** Provides more control for customization. [See Manual Installation](https://www.surfsense.net/docs/manual-installation)

**Prerequisites:** Ensure you complete the [prerequisite setup steps](https://www.surfsense.net/docs/) before installation, including:

*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

---

## Screenshots

**[Screenshots from Original README]**

---

## Tech Stack

**[Tech Stack Details from Original README]**

---

## Roadmap & Future Development

SurfSense is under active development with new features and improvements planned.

*   **Roadmap:** Stay updated on our progress via the [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   **Contribute:** Help shape the future of SurfSense by joining our [Discord community](https://discord.gg/ejRNvftDp9) and contributing.

---

## Contribute

We welcome contributions!  Small or large contributions (even a star!) are appreciated. For detailed guidelines, see our [CONTRIBUTING.md](CONTRIBUTING.md) file.

---

## Star History

**[Star History Chart from Original README]**