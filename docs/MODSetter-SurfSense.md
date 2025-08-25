![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

# SurfSense: Your Personalized AI Research Agent

**SurfSense empowers you to conduct in-depth research using your own knowledge base, integrating with various sources to provide insightful, cited answers.**  Explore the [SurfSense GitHub repository](https://github.com/MODSetter/SurfSense) for more details.

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)


## Key Features

*   💡 **Personalized AI Research**:  Create your own private, customizable research agent, like a private NotebookLM or Perplexity, powered by AI.
*   📁 **Extensive File Format Support**: Upload and save content from various file formats to your personal knowledge base, with support for 50+ file extensions via LlamaCloud.
*   🔍 **Advanced Search Capabilities**: Quickly and efficiently research and find information within your saved content.
*   💬 **Natural Language Interaction**: Engage in conversational interactions with your saved content, receiving cited answers for enhanced understanding.
*   📄 **Cited Answers**: Get reliable, cited answers to back up your research.
*   🔔 **Privacy-Focused and Local LLM Support**: Works seamlessly with local LLMs such as Ollama, ensuring privacy and flexibility.
*   🏠 **Self-Hosted & Open Source**: Easily deploy SurfSense locally for complete control over your data and research process.
*   🎙️ **AI-Powered Podcasts**:
    *   Blazingly fast podcast generation (under 20 seconds for a 3-minute podcast).
    *   Convert chat conversations into engaging audio content.
    *   Support for local TTS providers (Kokoro TTS).
    *   Support for multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Cutting-Edge RAG Techniques**:
    *   Supports 100+ LLMs.
    *   6000+ Embedding Models supported.
    *   All major Rerankers supported (Pinecone, Cohere, Flashrank, etc.).
    *   Hierarchical Indices (2-tiered RAG setup).
    *   Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **External Source Integration**: Connect to a wide range of external data sources, including:
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
    *   And more coming soon!
*   🔖 **Browser Extension**: Save any webpage content directly into your SurfSense knowledge base via a cross-browser extension.

## Supported File Extensions

> Note: File format support depends on your ETL service configuration (LlamaCloud, Unstructured, or Docling).

### Documents & Text
*   **LlamaCloud**: `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`
*   **Unstructured**: `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub`
*   **Docling**: `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc`

### Presentations
*   **LlamaCloud**: `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key`
*   **Unstructured**: `.ppt`, `.pptx`
*   **Docling**: `.pptx`

### Spreadsheets & Data
*   **LlamaCloud**: `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, `.123`, `.dif`, `.sylk`, `.slk`, `.prn`, `.et`, `.uos1`, `.uos2`, `.wk1`, `.wk2`, `.wk3`, `.wk4`, `.wks`, `.wq1`, `.wq2`, `.wb1`, `.wb2`, `.wb3`, `.qpw`, `.xlr`, `.eth`
*   **Unstructured**: `.xls`, `.xlsx`, `.csv`, `.tsv`
*   **Docling**: `.xlsx`, `.csv`

### Images
*   **LlamaCloud**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web`
*   **Unstructured**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`
*   **Docling**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

### Audio & Video (Always Supported)
*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication
*   **Unstructured**: `.eml`, `.msg`, `.p7s`

## Screenshots

*   **Research Agent**

    ![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)

*   **Search Spaces**

    ![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)

*   **Manage Documents**

    ![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)

*   **Podcast Agent**

    ![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)

*   **Agent Chat**

    ![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)

*   **Browser Extension**

    ![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)

    ![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### **Backend**

*   **FastAPI**: Fast, modern web framework.
*   **PostgreSQL with pgvector**: Database with vector search capabilities.
*   **SQLAlchemy**: ORM for database interactions.
*   **Alembic**: Database migrations tool.
*   **FastAPI Users**: Authentication and user management.
*   **LangGraph**: Framework for AI agents.
*   **LangChain**: Framework for AI-powered applications.
*   **LLM Integration**: Through LiteLLM.
*   **Rerankers**: For improved search relevance.
*   **Hybrid Search**: Combines vector and full-text search.
*   **Vector Embeddings**: For semantic search.
*   **pgvector**: PostgreSQL extension for vector operations.
*   **Chonkie**: Document chunking and embedding library.
    *   Uses `AutoEmbeddings` for flexible embedding model selection.
    *   `LateChunker` for optimized document chunking.

### **Frontend**

*   **Next.js 15.2.3**: React framework.
*   **React 19.0.0**: JavaScript library for user interfaces.
*   **TypeScript**: Static type-checking.
*   **Vercel AI SDK Kit UI Stream Protocol**: For scalable chat UI.
*   **Tailwind CSS 4.x**: Utility-first CSS framework.
*   **Shadcn**: Headless components library.
*   **Lucide React**: Icon set.
*   **Framer Motion**: Animation library.
*   **Sonner**: Toast notification library.
*   **Geist**: Font family.
*   **React Hook Form**: Form state management.
*   **Zod**: TypeScript-first schema validation.
*   **@hookform/resolvers**: Validation library resolvers.
*   **@tanstack/react-table**: Headless UI for tables & datagrids.

### **DevOps**

*   **Docker**: Container platform.
*   **Docker Compose**: Multi-container Docker applications.
*   **pgAdmin**: Web-based PostgreSQL administration tool.

### **Extension**
* Manifest v3 on Plasmo

## How to Get Started

SurfSense offers two installation options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - Easiest setup with containerization.
    *   Includes pgAdmin for database management.
    *   Supports environment variable customization.
    *   Flexible deployment options.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md).

2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For more control.

Before installation, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:

*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Roadmap & Future Work

*   [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   Add More Connectors
*   Patch minor bugs
*   Document Chat [REIMPLEMENT]
*   Document Podcasts

## Contribute

Contributions are welcome! See our [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

## Star History

<!-- Star History Chart -->
<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>