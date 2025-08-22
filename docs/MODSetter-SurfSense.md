<div align="center">
  <a href="https://github.com/MODSetter/SurfSense">
    <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
  </a>
</div>

<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord&style=flat-square" alt="Discord">
  </a>
</div>

# SurfSense: Your Customizable AI Research Assistant

**SurfSense empowers you to conduct comprehensive research by connecting your personal knowledge base to diverse external sources.**

[View the original repository on GitHub](https://github.com/MODSetter/SurfSense)

<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## Key Features

*   💡 **Personalized AI Research:** Build your private, customizable NotebookLM and Perplexity integrated with external sources.
*   📁 **Comprehensive File Support:** Upload and store content from various file formats, including documents, images, and videos, supporting 50+ file extensions.
*   🔍 **Advanced Search Capabilities:** Quickly find information within your saved content.
*   💬 **Interactive Chat Interface:** Converse with your saved content in natural language and receive cited answers.
*   📄 **Cited & Verified Answers:** Get answers cited just like Perplexity.
*   🔔 **Privacy-Focused & Local LLM Support:** Works seamlessly with local LLMs like Ollama.
*   🏠 **Self-Hosting Capabilities:** Open-source and easy to deploy locally.
*   🎙️ **AI-Powered Podcasts:**
    *   Blazingly fast podcast generation.
    *   Convert chat conversations into engaging audio content.
    *   Support for local and cloud-based TTS providers (OpenAI, Azure, Google Vertex AI, and Kokoro TTS).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs and 6000+ Embedding Models.
    *   Integrates with major rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes hierarchical indexing (2-tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion).
    *   Offers RAG-as-a-Service API backend.
*   ℹ️ **Extensive External Source Integrations:**
    *   Search Engines (Tavily, LinkUp)
    *   Slack
    *   Linear
    *   Jira
    *   ClickUp
    *   Confluence
    *   Notion
    *   YouTube Videos
    *   GitHub
    *   Discord
    *   And more to come...
*   🔖 **Cross-Browser Extension**: Save any webpage you like to your personal knowledge base.

## Supported File Extensions

> **Note**: File format support depends on your ETL service configuration. LlamaCloud supports 50+ formats, Unstructured supports 34+ core formats, and Docling (core formats, local processing, privacy-focused, no API key).

### Documents & Text

*   **LlamaCloud:** `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`
*   **Unstructured:** `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub`
*   **Docling:** `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc`

### Presentations

*   **LlamaCloud:** `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key`
*   **Unstructured:** `.ppt`, `.pptx`
*   **Docling:** `.pptx`

### Spreadsheets & Data

*   **LlamaCloud:** `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, `.123`, `.dif`, `.sylk`, `.slk`, `.prn`, `.et`, `.uos1`, `.uos2`, `.wk1`, `.wk2`, `.wk3`, `.wk4`, `.wks`, `.wq1`, `.wq2`, `.wb1`, `.wb2`, `.wb3`, `.qpw`, `.xlr`, `.eth`
*   **Unstructured:** `.xls`, `.xlsx`, `.csv`, `.tsv`
*   **Docling:** `.xlsx`, `.csv`

### Images

*   **LlamaCloud:** `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web`
*   **Unstructured:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`
*   **Docling:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

### Audio & Video *(Always Supported)*

*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured:** `.eml`, `.msg`, `.p7s`

## Installation

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest way to get SurfSense up and running with all dependencies containerized.
    *   Includes pgAdmin for database management through a web UI
    *   Supports environment variable customization via `.env` file
    *   Flexible deployment options (full stack or core services only)
    *   No need to manually edit configuration files between environments
    *   See [Docker Setup Guide](DOCKER_SETUP.md) for detailed instructions
    *   For deployment scenarios and options, see [Deployment Guide](DEPLOYMENT_GUIDE.md)

2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For users who prefer more control over their setup or need to customize their deployment.

Before installation, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:

*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Screenshots

**Research Agent**

![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)

**Search Spaces**

![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)

**Manage Documents**

![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)

**Podcast Agent**

![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)

**Agent Chat**

![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)

**Browser Extension**

![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)

![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### Backend

*   **FastAPI:** Web framework for building APIs with Python
*   **PostgreSQL with pgvector:** Database with vector search capabilities
*   **SQLAlchemy:** SQL toolkit and ORM for database interactions
*   **Alembic:** Database migrations tool
*   **FastAPI Users:** Authentication and user management with JWT and OAuth support
*   **LangGraph:** Framework for developing AI-agents.
*   **LangChain:** Framework for developing AI-powered applications.
*   **LLM Integration:** Integration with LLM models through LiteLLM
*   **Rerankers:** Advanced result ranking for improved search relevance
*   **Hybrid Search:** Combines vector similarity and full-text search using Reciprocal Rank Fusion (RRF)
*   **Vector Embeddings:** Document and text embeddings for semantic search
*   **pgvector:** PostgreSQL extension for vector similarity operations
*   **Chonkie:** Advanced document chunking and embedding library
    *   Uses `AutoEmbeddings` for flexible embedding model selection
    *   `LateChunker` for optimized document chunking

### Frontend

*   **Next.js 15.2.3:** React framework with features like App Router and server components.
*   **React 19.0.0:** JavaScript library for building user interfaces.
*   **TypeScript:** Static type-checking for JavaScript.
*   **Vercel AI SDK Kit UI Stream Protocol:** For scalable chat UI.
*   **Tailwind CSS 4.x:** Utility-first CSS framework.
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set implemented as React components.
*   **Framer Motion:** Animation library for React.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family from Vercel.
*   **React Hook Form:** Form state management and validation.
*   **Zod:** TypeScript-first schema validation.
*   **@hookform/resolvers:** Resolvers for using validation libraries with React Hook Form.
*   **@tanstack/react-table:** Headless UI for building tables.

### DevOps

*   **Docker:** Container platform.
*   **Docker Compose:** Tool for multi-container Docker applications.
*   **pgAdmin:** Web-based PostgreSQL administration tool (included in Docker setup).

### Extension

*   Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Chat **[REIMPLEMENT]**
*   Document Podcasts

## Contribute

Contributions are welcomed! [Read the Contributing Guidelines](CONTRIBUTING.md) for details.

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
    <picture>
        <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
        <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
        <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
    </picture>
</a>