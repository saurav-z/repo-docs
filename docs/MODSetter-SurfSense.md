# SurfSense: Supercharge Your Research with an AI-Powered Knowledge Hub

**Tired of information overload? SurfSense empowers you to build a personalized AI research assistant by connecting your knowledge base to various sources and tools.** ([Back to the Repo](https://github.com/MODSetter/SurfSense))

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

[Video Demo](https://github.com/user-attachments/assets/d9221908-e0de-4b2f-ac3a-691cf4b202da)

[Podcast Sample](https://github.com/user-attachments/assets/a0a16566-6967-4374-ac51-9b3e07fbecd7)

## Key Features

*   💡 **AI-Powered Knowledge Base**: Create your own private, customizable NotebookLM and Perplexity, integrated with your personal data sources.
*   📁 **Extensive File Support**: Upload and manage content from various formats, including documents, images, videos, and more (supports 50+ file extensions).
*   🔍 **Powerful Search**: Quickly find information within your saved content.
*   💬 **Conversational Interaction**: Chat with your data using natural language and receive cited answers, just like Perplexity.
*   🔔 **Privacy & Local LLM Support**: Works seamlessly with local LLMs like Ollama.
*   🏠 **Self-Hostable**: Open-source and easy to deploy locally.
*   🎙️ **AI-Powered Podcasts**:
    *   Blazingly fast podcast generation agent. (Creates a 3-minute podcast in under 20 seconds.)
    *   Convert chat conversations into engaging audio content.
    *   Support for local TTS providers (Kokoro TTS).
    *   Support for multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques**:
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecode, Cohere, Flashrank, etc.).
    *   Uses Hierarchical Indices (2 tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **External Source Integrations**:
    *   Search Engines (Tavily, LinkUp)
    *   Slack
    *   Linear
    *   Jira
    *   ClickUp
    *   Confluence
    *   Notion
    *   Gmail
    *   YouTube Videos
    *   GitHub
    *   Discord
    *   Google Calendar
    *   And More to Come!
*   🔖 **Cross-Browser Extension**: Save webpages directly into your knowledge base using the SurfSense browser extension.

## Supported File Extensions

File format support depends on your ETL service configuration.  Consider these options:

*   **LlamaCloud:** Supports 50+ formats
*   **Unstructured:** Supports 34+ core formats
*   **Docling:** Privacy-focused local processing with no API key; supports core formats.

### Documents & Text
**LlamaCloud**: `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`

**Unstructured**: `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub`

**Docling**: `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc`

### Presentations
**LlamaCloud**: `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key`

**Unstructured**: `.ppt`, `.pptx`

**Docling**: `.pptx`

### Spreadsheets & Data
**LlamaCloud**: `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, `.123`, `.dif`, `.sylk`, `.slk`, `.prn`, `.et`, `.uos1`, `.uos2`, `.wk1`, `.wk2`, `.wk3`, `.wk4`, `.wks`, `.wq1`, `.wq2`, `.wb1`, `.wb2`, `.wb3`, `.qpw`, `.xlr`, `.eth`

**Unstructured**: `.xls`, `.xlsx`, `.csv`, `.tsv`

**Docling**: `.xlsx`, `.csv`

### Images
**LlamaCloud**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web`

**Unstructured**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`

**Docling**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

### Audio & Video *(Always Supported)*
`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication
**Unstructured**: `.eml`, `.msg`, `.p7s`

## Get Started

### Installation Options

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** (Recommended):  The easiest way to deploy with all dependencies containerized, including pgAdmin for database management through a web UI.  Supports environment variable customization, flexible deployment options, and simplifies environment-specific configuration.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md)
2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)**: For users who prefer more control.

Before installing, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:
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

### **Backend**

*   **FastAPI**: Modern, fast web framework for building APIs with Python
*   **PostgreSQL with pgvector**: Database with vector search capabilities for similarity searches
*   **SQLAlchemy**: SQL toolkit and ORM (Object-Relational Mapping) for database interactions
*   **Alembic**: A database migrations tool for SQLAlchemy.
*   **FastAPI Users**: Authentication and user management with JWT and OAuth support
*   **LangGraph**: Framework for developing AI-agents.
*   **LangChain**: Framework for developing AI-powered applications.
*   **LLM Integration**: Integration with LLM models through LiteLLM
*   **Rerankers**: Advanced result ranking for improved search relevance
*   **Hybrid Search**: Combines vector similarity and full-text search for optimal results using Reciprocal Rank Fusion (RRF)
*   **Vector Embeddings**: Document and text embeddings for semantic search
*   **pgvector**: PostgreSQL extension for efficient vector similarity operations
*   **Chonkie**: Advanced document chunking and embedding library
    *   Uses `AutoEmbeddings` for flexible embedding model selection
    *   `LateChunker` for optimized document chunking based on embedding model's max sequence length

### **Frontend**

*   **Next.js 15.2.3**: React framework featuring App Router, server components, automatic code-splitting, and optimized rendering.
*   **React 19.0.0**: JavaScript library for building user interfaces.
*   **TypeScript**: Static type-checking for JavaScript, enhancing code quality and developer experience.
*   **Vercel AI SDK Kit UI Stream Protocol**: To create scalable chat UI.
*   **Tailwind CSS 4.x**: Utility-first CSS framework for building custom UI designs.
*   **Shadcn**: Headless components library.
*   **Lucide React**: Icon set implemented as React components.
*   **Framer Motion**: Animation library for React.
*   **Sonner**: Toast notification library.
*   **Geist**: Font family from Vercel.
*   **React Hook Form**: Form state management and validation.
*   **Zod**: TypeScript-first schema validation with static type inference.
*   **@hookform/resolvers**: Resolvers for using validation libraries with React Hook Form.
*   **@tanstack/react-table**: Headless UI for building powerful tables & datagrids.

### **DevOps**

*   **Docker**: Container platform for consistent deployment across environments
*   **Docker Compose**: Tool for defining and running multi-container Docker applications
*   **pgAdmin**: Web-based PostgreSQL administration tool included in Docker setup

### **Extension**
 Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Chat **[REIMPLEMENT]**
*   Document Podcasts

## Contribute

Contributions are very welcome!  Even a ⭐ helps!

For detailed contribution guidelines, see [CONTRIBUTING.md](CONTRIBUTING.md).

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
    <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
    <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
  </picture>
</a>