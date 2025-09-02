<div align="center">
  <a href="https://github.com/MODSetter/SurfSense" target="_blank">
    <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header" width="100%">
  </a>
</div>

<div align="center">
  <a href="https://discord.gg/ejRNvftDp9" target="_blank">
    <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord&style=flat-square" alt="Discord">
  </a>
</div>

# SurfSense: Your AI-Powered Research Assistant, Connecting to Your Knowledge Base

SurfSense revolutionizes research by connecting your personal knowledge with external sources, offering a highly customizable AI research agent. [**Explore SurfSense on GitHub!**](https://github.com/MODSetter/SurfSense)

<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## Key Features

*   **Private Knowledge Base Integration:** Turn your saved content (documents, images, videos, and 50+ file formats) into your own searchable and conversational knowledge base.
*   **Advanced Search Capabilities:** Quickly find information within your saved content.
*   **AI-Powered Chat:** Interact with your saved content using natural language and receive cited answers, just like Perplexity.
*   **Local LLM Support:** Works seamlessly with local LLMs like Ollama for enhanced privacy.
*   **Self-Hosted Solution:** Open-source and easy to deploy locally, giving you complete control.
*   **Podcast Generation:** Generate engaging audio content from your conversations, including a blazingly fast podcast generation agent that creates 3-minute podcasts in under 20 seconds.
*   **Advanced RAG Techniques:** Leverages cutting-edge Retrieval-Augmented Generation (RAG) techniques:
    *   Supports 100+ LLMs.
    *   Supports 6000+ embedding models.
    *   Utilizes all major rerankers.
    *   Employs a two-tiered hierarchical indexing system.
    *   Combines Semantic and Full Text Search for hybrid search results with Reciprocal Rank Fusion.
    *   RAG as a Service API Backend.
*   **Extensive External Source Integration:** Connects to a wide range of sources, including:
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
    *   And more to come!
*   **Cross-Browser Extension:** Save any webpage with the SurfSense browser extension (Manifest v3 on Plasmo).

## Supported File Extensions

SurfSense supports a wide array of file formats. The supported formats depend on the ETL service configuration (LlamaCloud, Unstructured, or Docling).

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

### Audio & Video (Always Supported)

*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured:** `.eml`, `.msg`, `.p7s`

---

## Getting Started

SurfSense offers two primary installation methods:

1.  **Docker Installation:** The easiest way to get SurfSense up and running with all dependencies containerized. Includes pgAdmin for database management through a web UI and supports environment variable customization via `.env` file. See the [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md).
2.  **Manual Installation (Recommended):** For users who prefer more control.  Detailed OS-specific instructions are available in the manual installation guides.

Before installation, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including PGVector setup, and choosing a File Processing ETL Service and other required API keys.

## Screenshots

**(Images of the research agent, search spaces, document management, podcast agent, agent chat, and browser extension would be displayed here)**

---

## Tech Stack

### Backend

*   **FastAPI:** Web framework for building APIs.
*   **PostgreSQL with pgvector:** Database with vector search capabilities.
*   **SQLAlchemy:** SQL toolkit and ORM for database interactions.
*   **Alembic:** Database migrations tool.
*   **FastAPI Users:** Authentication and user management.
*   **LangGraph & LangChain:** Frameworks for AI agent development.
*   **LLM Integration:** LiteLLM for integrating with LLM models.
*   **Rerankers & Hybrid Search:** Advanced search result ranking.
*   **Vector Embeddings:** Document and text embeddings.
*   **pgvector:** PostgreSQL extension for vector operations.
*   **Chonkie:** Advanced document chunking library.

### Frontend

*   **Next.js:** React framework with App Router.
*   **React:** JavaScript library for building user interfaces.
*   **TypeScript:** Static type-checking.
*   **Vercel AI SDK Kit UI Stream Protocol:** For scalable chat UI.
*   **Tailwind CSS:** Utility-first CSS framework.
*   **Shadcn & Lucide React:** Component and icon libraries.
*   **Framer Motion:** Animation library.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family from Vercel.
*   **React Hook Form & Zod:** Form state management and validation.
*   **@hookform/resolvers:** For using validation libraries with React Hook Form.
*   **@tanstack/react-table:** UI for building tables.

### DevOps

*   **Docker & Docker Compose:** Containerization and orchestration.
*   **pgAdmin:** Web-based PostgreSQL administration.

### Extension

*   Manifest v3 on Plasmo

---

## Contribute

Contributions are welcomed!  See the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.

## Roadmap & Future

SurfSense is under active development. Explore the [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2).

Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to get involved!

## Star History

**(Include Star History Chart Here - see original repo for link)**

---

**(Replace placeholder image URLs with the actual URLs)**