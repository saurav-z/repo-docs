[![Discord](https://img.shields.io/discord/1359368468260192417?logo=discord&label=Discord)](https://discord.gg/ejRNvftDp9)

# SurfSense: Your Customizable AI Research Agent

**SurfSense revolutionizes research by connecting your personal knowledge base to a wide array of external sources, providing powerful search and insightful answers.** [Check out the original repo](https://github.com/MODSetter/SurfSense)!

[<img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/13606)

[![SurfSense Demo Video](https://img.youtube.com/vi/YOUR_YOUTUBE_VIDEO_ID/0.jpg)](https://github.com/user-attachments/assets/d9221908-e0de-4b2f-ac3a-691cf4b202da) (Replace "YOUR_YOUTUBE_VIDEO_ID" with the actual ID.)

## Key Features

*   **Private Knowledge Base Integration:** Connects to your files and various sources for personalized research.
*   **Multiple File Format Support:** Upload and save content from documents, images, videos, and more, with support for **50+ file extensions**.
*   **Powerful Search Capabilities:** Quickly find anything within your saved content.
*   **Interactive Chat Interface:** Engage in natural language conversations and receive cited answers.
*   **Cited Answers:** Get reliable answers with source citations, just like Perplexity.
*   **Privacy-Focused & Local LLM Support:** Works seamlessly with Ollama and other local LLMs.
*   **Self-Hostable:** Open source and easy to deploy locally.
*   **Podcast Generation Agent:**
    *   Generate podcasts from conversations (3-minute podcast in under 20 seconds).
    *   Supports local TTS providers (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Uses Hierarchical Indices (2-tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **External Source Integrations:**
    *   Search Engines: Tavily, LinkUp.
    *   Collaboration Tools: Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar.
    *   And more to come!
*   **Browser Extension:** Save web pages directly into your knowledge base.

## Supported File Extensions

*   Detailed file format support varies based on your chosen ETL service (LlamaCloud, Unstructured, Docling). Check the documentation for details.

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

### Audio & Video *(Always Supported)*
`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication
*   **Unstructured**: `.eml`, `.msg`, `.p7s`

## Screenshots

*   **Research Agent**: ![Research Agent](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
*   **Search Spaces**: ![Search Spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
*   **Manage Documents**: ![Manage Documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)
*   **Podcast Agent**: ![Podcast Agent](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)
*   **Agent Chat**: ![Agent Chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)
*   **Browser Extension**:
    *   ![Extension 1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)
    *   ![Extension 2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### Backend

*   **FastAPI**: Fast web framework for building APIs with Python
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

### Frontend

*   **Next.js 15.2.3**: React framework
*   **React 19.0.0**: JavaScript library for building user interfaces
*   **TypeScript**: Static type-checking for JavaScript
*   **Vercel AI SDK Kit UI Stream Protocol**: To create scalable chat UI
*   **Tailwind CSS 4.x**: Utility-first CSS framework
*   **Shadcn**: Headless components library
*   **Lucide React**: Icon set implemented as React components
*   **Framer Motion**: Animation library for React
*   **Sonner**: Toast notification library
*   **Geist**: Font family from Vercel
*   **React Hook Form**: Form state management and validation
*   **Zod**: TypeScript-first schema validation with static type inference
*   **@hookform/resolvers**: Resolvers for using validation libraries with React Hook Form
*   **@tanstack/react-table**: Headless UI for building powerful tables & datagrids

### DevOps

*   **Docker**: Container platform
*   **Docker Compose**: Tool for defining and running multi-container Docker applications
*   **pgAdmin**: Web-based PostgreSQL administration tool

### Extension

*   Manifest v3 on Plasmo

## Installation

### Installation Options

SurfSense provides two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest way to get SurfSense up and running.
    *   Includes pgAdmin for database management
    *   Supports environment variable customization via `.env` file
    *   Flexible deployment options (full stack or core services only)
    *   No need to manually edit configuration files
    *   See [Docker Setup Guide](DOCKER_SETUP.md)
    *   For deployment scenarios and options, see [Deployment Guide](DEPLOYMENT_GUIDE.md)

2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For users who prefer more control.

Both installation guides include detailed OS-specific instructions for Windows, macOS, and Linux.

Before installation, complete the [prerequisite setup steps](https://www.surfsense.net/docs/) including:
*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Future Work

*   Add More Connectors
*   Patch minor bugs
*   Document Podcasts

## Contribute

Contributions are very welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Roadmap & Community

*   **Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   **Join the Community:** [SurfSense Discord](https://discord.gg/ejRNvftDp9) for feature requests, feedback, and discussions.

## Star History

[![Star History](https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date)](https://star-history.com/#MODSetter/SurfSense&Date)

---
---
<p align="center">
    <img
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4"
      alt="Catalyst Project"
      width="200"
    />
</p>