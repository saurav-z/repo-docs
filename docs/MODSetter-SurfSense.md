<div align="center">
<a href="https://github.com/MODSetter/SurfSense">
  <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
</a>
</div>

<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">
  </a>
</div>

# SurfSense: Your Customizable AI Research Agent 🚀

**SurfSense transforms your research process by connecting your personal knowledge base to external sources, offering a powerful, open-source AI research assistant.**

[View the original repository on GitHub](https://github.com/MODSetter/SurfSense)

---

## Key Features

*   **Private & Customizable AI Research:** Build your own personalized research hub, similar to NotebookLM or Perplexity, but connected to your data.
*   **Connect to Your Data:**
    *   Upload files in **50+ formats**: Documents, images, videos, and more.
    *   Integrate with a wide range of sources: Search engines (Tavily, LinkUp), Slack, Linear, Jira, ClickUp, Confluence, Gmail, Notion, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.
*   **Powerful Search & Chat:**
    *   Quickly find anything in your saved content with advanced search capabilities.
    *   Engage in natural language conversations to get cited answers.
*   **Privacy & Control:**
    *   Supports local LLMs like Ollama, ensuring privacy.
    *   Self-hostable for full control over your data.
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs, 6000+ Embedding Models, and all major Rerankers (Pinecode, Cohere, Flashrank).
    *   Employs Hierarchical Indices and Hybrid Search (Semantic + Full Text Search) for optimal results.
    *   RAG as a Service API Backend.
*   **Podcast Agent:**
    *   Blazingly fast podcast generation (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into engaging audio content.
    *   Support for local and multiple TTS providers.
*   **Browser Extension:** Save any webpage.

---

## Supported File Extensions

SurfSense supports a wide variety of file formats.

*   **Documents & Text:**
    *   **LlamaCloud:** `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`
    *   **Unstructured:** `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub`
    *   **Docling:** `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc`
*   **Presentations:**
    *   **LlamaCloud:** `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key`
    *   **Unstructured:** `.ppt`, `.pptx`
    *   **Docling:** `.pptx`
*   **Spreadsheets & Data:**
    *   **LlamaCloud:** `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, `.123`, `.dif`, `.sylk`, `.slk`, `.prn`, `.et`, `.uos1`, `.uos2`, `.wk1`, `.wk2`, `.wk3`, `.wk4`, `.wks`, `.wq1`, `.wq2`, `.wb1`, `.wb2`, `.wb3`, `.qpw`, `.xlr`, `.eth`
    *   **Unstructured:** `.xls`, `.xlsx`, `.csv`, `.tsv`
    *   **Docling:** `.xlsx`, `.csv`
*   **Images:**
    *   **LlamaCloud:** `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web`
    *   **Unstructured:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`
    *   **Docling:** `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`
*   **Audio & Video:** `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`
*   **Email & Communication:**
    *   **Unstructured:** `.eml`, `.msg`, `.p7s`

---

## Installation

SurfSense offers flexible installation options.

*   **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: Easiest setup with containerized dependencies and environment variable customization.
*   **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)**: For users requiring more control.

**Prerequisites:**  PGVector setup, ETL service configuration (Unstructured.io, LlamaIndex, or Docling), and required API keys.  Refer to the detailed installation guides for OS-specific instructions.

---

## Technology Stack

### Backend

*   **FastAPI**: API Framework
*   **PostgreSQL with pgvector**: Database with vector search
*   **SQLAlchemy**: Database ORM
*   **Alembic**: Database Migrations
*   **FastAPI Users**: User Management & Authentication
*   **LangGraph**: AI-Agent Framework
*   **LangChain**: AI Application Framework
*   **LLM Integration**: LiteLLM
*   **Rerankers**: Result Ranking
*   **Hybrid Search**: Vector & Full Text Search
*   **Vector Embeddings**: Semantic Search
*   **pgvector**: PostgreSQL Extension
*   **Chonkie**: Document chunking and embedding

### Frontend

*   **Next.js**: React Framework
*   **React**: UI Library
*   **TypeScript**: Type-checking
*   **Vercel AI SDK Kit UI Stream Protocol**: Chat UI
*   **Tailwind CSS**: UI Framework
*   **Shadcn**: UI Components
*   **Lucide React**: Icon Set
*   **Framer Motion**: Animation
*   **Sonner**: Toast Notifications
*   **Geist**: Font Family
*   **React Hook Form**: Form Management
*   **Zod**: Schema Validation
*   **@hookform/resolvers**: Validation Resolvers
*   **@tanstack/react-table**: Table & DataGrid

### DevOps

*   **Docker**: Containerization
*   **Docker Compose**: Multi-container Application
*   **pgAdmin**: PostgreSQL Admin Tool

### Extension

*   Manifest v3 on Plasmo

---

## Roadmap & Community

Stay up to date with the latest developments.

*   **Roadmap:**  [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   **Join the Community:** Get involved and shape the future of SurfSense in the [SurfSense Discord](https://discord.gg/ejRNvftDp9).

---

## Contribute

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.  Stars are appreciated!

---

## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date)](https://star-history.com/#MODSetter/SurfSense&Date)

---

<p align="center">
    <img 
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4" 
      alt="Catalyst Project" 
      width="200"
    />
</p>