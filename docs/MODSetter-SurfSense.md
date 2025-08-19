# SurfSense: Your AI-Powered Research Assistant

**Tired of information overload? SurfSense transforms your research process by connecting to your personal knowledge base and external sources, making complex research simple.** ([Original Repo](https://github.com/MODSetter/SurfSense))

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord)](https://discord.gg/ejRNvftDp9)

[<img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **AI-Powered Knowledge Base:** Create a private, customizable NotebookLM and Perplexity experience with your own data and external sources.
*   📁 **Extensive File Support:** Upload and manage content from various file formats, supporting 50+ file extensions.
*   🔍 **Intelligent Search:** Quickly find information within your saved content.
*   💬 **Conversational AI:** Interact with your data using natural language and get cited answers.
*   📄 **Cited Answers:** Receive answers with source citations, just like Perplexity.
*   🔔 **Privacy & Local LLM Support:** Seamlessly works with Ollama local LLMs.
*   🏠 **Self-Hosted:** Open source and easy to deploy locally.
*   🎙️ **Podcast Generation:**
    *   Blazingly fast podcast creation (under 20 seconds for a 3-minute podcast).
    *   Convert chat conversations to audio.
    *   Support for local (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers.
    *   Hierarchical Indices (2-tiered RAG setup).
    *   Hybrid Search (Semantic + Full Text Search combined with RRF).
    *   RAG as a Service API Backend.
*   ℹ️ **External Source Integrations:**
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
    *   And more coming soon!
*   🌐 **Cross-Browser Extension:** Save any webpage directly to your knowledge base with our extension.

## Supported File Extensions

The types of files you can use depend on the ETL service you are using.

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

## Installation

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: Easiest method with all dependencies containerized. Includes pgAdmin. Environment variables can be configured via `.env` files. See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md) for more details.

2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)**: More control for advanced users.

**Prerequisites:**

*   PGVector setup
*   File Processing ETL Service (choose one and obtain API key):
    *   Unstructured.io (34+ formats)
    *   LlamaIndex (50+ formats)
    *   Docling (local processing)
*   Other required API keys

## Screenshots

### Research Agent

![Research Agent](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)

### Search Spaces

![Search Spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)

### Manage Documents

![Manage Documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)

### Podcast Agent

![Podcast Agent](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)

### Agent Chat

![Agent Chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)

### Browser Extension

![Browser Extension 1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)

![Browser Extension 2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### **BackEnd**

*   **FastAPI**: Modern, fast web framework for building APIs with Python
*   **PostgreSQL with pgvector**: Database with vector search capabilities for similarity searches
*   **SQLAlchemy**: SQL toolkit and ORM (Object-Relational Mapping) for database interactions
*   **Alembic**: A database migrations tool for SQLAlchemy
*   **FastAPI Users**: Authentication and user management with JWT and OAuth support
*   **LangGraph**: Framework for developing AI-agents
*   **LangChain**: Framework for developing AI-powered applications
*   **LLM Integration**: Integration with LLM models through LiteLLM
*   **Rerankers**: Advanced result ranking for improved search relevance
*   **Hybrid Search**: Combines vector similarity and full-text search for optimal results using Reciprocal Rank Fusion (RRF)
*   **Vector Embeddings**: Document and text embeddings for semantic search
*   **pgvector**: PostgreSQL extension for efficient vector similarity operations
*   **Chonkie**: Advanced document chunking and embedding library
    *   Uses `AutoEmbeddings` for flexible embedding model selection
    *   `LateChunker` for optimized document chunking based on embedding model's max sequence length

### **FrontEnd**

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

*   Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Chat **[REIMPLEMENT]**
*   Document Podcasts

## Contribute

Contributions are very welcome! Star the repo or contribute code. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>