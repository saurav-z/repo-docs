# SurfSense: Your Customizable AI Research Agent

**SurfSense empowers you to conduct research and manage information seamlessly by integrating your personal knowledge base with powerful AI tools.** ([Back to Top](https://github.com/MODSetter/SurfSense))

[<img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">](https://discord.gg/ejRNvftDp9)

[<img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **AI-Powered Knowledge Management:** Build a private, customizable research hub akin to NotebookLM or Perplexity, integrated with your data.
*   📁 **Multi-Format File Support:** Upload and store content from various file types, including documents, images, videos, and more (supporting 50+ file extensions).
*   🔍 **Advanced Search Capabilities:** Quickly find specific information within your stored content using powerful search tools.
*   💬 **Conversational Interaction:** Engage in natural language conversations to get cited answers from your saved data.
*   📄 **Cited Answers:** Receive answers with source citations for easy verification and research.
*   🔔 **Privacy-Focused and Local LLM Support:** Works seamlessly with local LLMs like Ollama, ensuring data privacy.
*   🏠 **Self-Hosting Capabilities:** Open-source and easy to deploy locally for full control over your data.
*   🎙️ **Podcast Generation Agent:**
    *   Rapid podcast creation (3-minute podcasts in under 20 seconds).
    *   Convert chat conversations into audio content.
    *   Supports local and multiple TTS providers (e.g., OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs and 6000+ embedding models.
    *   Integrates with major rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes hierarchical indices (two-tiered RAG setup).
    *   Employs hybrid search (semantic + full-text search with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **Extensive External Source Integration:** Connect to various external sources, including:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.
*   🌐 **Browser Extension:** Save webpages directly to your SurfSense knowledge base.

## Supported File Extensions

SurfSense supports a wide array of file formats. The specific formats available depend on your chosen ETL service configuration (e.g., LlamaCloud, Unstructured.io, Docling).

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

`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured**: `.eml`, `.msg`, `.p7s`

## Installation

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - Easiest for quick setup, with all dependencies containerized. Includes pgAdmin for database management and supports environment variable customization.
    *   See the [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md) for details.
2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For more control and customization.

Before installing, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), which include PGVector and choosing a File Processing ETL service (e.g., Unstructured.io, LlamaIndex, Docling).

## Screenshots

*   **Research Agent**:  
    ![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
*   **Search Spaces**:  
    ![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
*   **Manage Documents**:  
    ![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)
*   **Podcast Agent**:  
    ![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)
*   **Agent Chat**:  
    ![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)
*   **Browser Extension**:  
    ![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)
    ![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### Backend

*   **FastAPI:** Web framework for building APIs.
*   **PostgreSQL with pgvector:** Database with vector search capabilities.
*   **SQLAlchemy:** SQL toolkit and ORM.
*   **Alembic:** Database migrations.
*   **FastAPI Users:** User management with JWT and OAuth.
*   **LangGraph:** AI-agent development framework.
*   **LangChain:** AI-powered application development.
*   **LLM Integration:** Through LiteLLM.
*   **Rerankers:** Advanced result ranking.
*   **Hybrid Search:** Combines vector similarity and full-text search.
*   **Vector Embeddings:** For semantic search.
*   **pgvector:** PostgreSQL extension for vector similarity.
*   **Chonkie:** Document chunking and embedding.
    *   Uses `AutoEmbeddings` for flexible embedding model selection.
    *   `LateChunker` for optimized document chunking.

### Frontend

*   **Next.js:** React framework.
*   **React:** Library for building user interfaces.
*   **TypeScript:** Type-checking.
*   **Vercel AI SDK Kit UI Stream Protocol:** Scalable chat UI.
*   **Tailwind CSS:** CSS framework.
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set.
*   **Framer Motion:** Animation library.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family.
*   **React Hook Form:** Form state management.
*   **Zod:** Schema validation.
*   **@hookform/resolvers:** Validation library resolvers.
*   **@tanstack/react-table:** Tables & datagrids.

### DevOps

*   **Docker:** Container platform.
*   **Docker Compose:** Multi-container applications.
*   **pgAdmin:** Web-based PostgreSQL administration.

### Extension

*   Manifest v3 on Plasmo

## Future Work

*   Add more connectors.
*   Patch minor bugs.
*   Document Podcasts.

## Contribute

Contributions are welcome! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>

---

<p align="center">
    <img 
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4" 
      alt="Catalyst Project" 
      width="200"
    />
</p>

---