[![Discord](https://img.shields.io/discord/1359368468260192417?style=social)](https://discord.gg/ejRNvftDp9)

# SurfSense: Your AI-Powered Research Companion

**SurfSense empowers you to conduct comprehensive research by connecting to your personal knowledge base and external sources, acting as your own private, highly customizable AI research agent.**

[View the original repository on GitHub](https://github.com/MODSetter/SurfSense)

[<img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/13606)

## Key Features:

*   **Private AI Research Hub:**  Integrate your personal files and external sources, like NotebookLM and Perplexity, but fully customizable.
*   **Multiple File Format Support:** Upload and save content from documents, images, videos, and more.  Supports **50+ file extensions**.
*   **Powerful Search:** Quickly research and find information within your saved content.
*   **AI-Powered Chat:** Interact with your saved content through natural language and receive cited answers.
*   **Cited Answers:**  Get answers with citations, similar to Perplexity.
*   **Local LLM Support:**  Works seamlessly with local LLMs like Ollama for enhanced privacy and control.
*   **Self-Hostable:** Open-source and easy to deploy locally.
*   **Podcast Generation:**
    *   Blazingly fast podcast creation (under 20 seconds for a 3-minute podcast).
    *   Convert chat conversations into audio content.
    *   Support for local TTS providers (e.g., Kokoro TTS) and multiple TTS providers (e.g., OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecode, Cohere, Flashrank etc.).
    *   Utilizes Hierarchical Indices (2-tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **External Source Integrations:**
    *   Search Engines: Tavily, LinkUp
    *   Collaboration Tools: Slack, Linear, Jira, ClickUp, Confluence, Discord, Airtable, Google Calendar
    *   Communication: Gmail
    *   Content Platforms: Notion, YouTube, GitHub
    *   *And more to come...*
*   **Browser Extension:**  Save any webpage, even behind authentication, with the SurfSense extension.

## Supported File Extensions:

SurfSense supports various file formats through integration with different ETL services. The level of support varies depending on the specific ETL service. Here's a breakdown:

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

## Getting Started:

### Installation Options:

SurfSense offers two installation paths:

1.  **Docker Installation:** The simplest way to get started, with all dependencies containerized. Includes pgAdmin for database management via a web UI and supports environment variable customization.  See the detailed [Docker Setup Guide](https://www.surfsense.net/docs/docker-installation).
2.  **Manual Installation (Recommended):** Provides greater control and customization. Refer to the [Manual Installation Guide](https://www.surfsense.net/docs/manual-installation) for instructions.

**Important:** Before installation, complete the prerequisite setup steps, including:

*   PGVector setup
*   File Processing ETL Service:
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Screenshots

**(Insert Screenshots here - replace placeholder with actual image links)**

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


## Tech Stack:

### Backend:

*   FastAPI
*   PostgreSQL with pgvector
*   SQLAlchemy
*   Alembic
*   FastAPI Users
*   LangGraph & LangChain
*   LLM Integration (LiteLLM)
*   Rerankers
*   Hybrid Search
*   Vector Embeddings
*   pgvector
*   Chonkie

### Frontend:

*   Next.js
*   React
*   TypeScript
*   Vercel AI SDK Kit UI Stream Protocol
*   Tailwind CSS
*   Shadcn
*   Lucide React
*   Framer Motion
*   Sonner
*   Geist
*   React Hook Form
*   Zod
*   @hookform/resolvers
*   @tanstack/react-table

### DevOps:

*   Docker
*   Docker Compose
*   pgAdmin

### Extension:
*   Manifest v3 on Plasmo

## Future Work:

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute:

Contributions are highly encouraged! Star the repo, open issues, or submit pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Star History

```html
<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>
```

---
---
<p align="center">
    <img
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4"
      alt="Catalyst Project"
      width="200"
    />
</p>

---