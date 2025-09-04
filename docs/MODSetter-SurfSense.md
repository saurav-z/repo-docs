# SurfSense: Your AI-Powered Research Assistant 🚀

SurfSense empowers you to conduct in-depth research by integrating your personal knowledge base with a wide array of external sources, including search engines, cloud services, and more.  [Check out the original repo](https://github.com/MODSetter/SurfSense).

<!--  Discord link -->
<div align="center">
<a href="https://discord.gg/ejRNvftDp9">
<img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">
</a>
</div>

<!-- Trendshift badge -->
<div align="center">
<a href="https://trendshift.io/repositories/13606" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>


## Key Features

*   **Personalized AI Research Hub:** Create your own private, customizable research environment, combining the capabilities of tools like NotebookLM and Perplexity.
*   **Comprehensive File Uploading:** Upload and organize content from various file formats, with support for 50+ file extensions through the LlamaCloud service.
*   **Powerful Search & Discovery:** Quickly search and find information within your saved content.
*   **Intelligent Chat Interface:**  Interact with your data through natural language, receiving cited answers for enhanced accuracy.
*   **Cited Answers:** Get answers with citations, just like Perplexity
*   **Local LLM Support:** Works seamlessly with local LLMs such as Ollama, promoting data privacy.
*   **Self-Hosted & Open Source:** Easy to deploy and customize locally.
*   **Podcast Generation:**
    *   Blazingly fast podcast generation agent. (Creates a 3-minute podcast in under 20 seconds.)
    *   Convert your chat conversations into engaging audio content
    *   Support for local TTS providers (Kokoro TTS)
    *   Support for multiple TTS providers (OpenAI, Azure, Google Vertex AI)
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecode, Cohere, Flashrank etc)
    *   Uses Hierarchical Indices (2 tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **External Data Integration:** Connect to diverse data sources:
    *   Search Engines (Tavily, LinkUp)
    *   Slack
    *   Linear
    *   Jira
    *   ClickUp
    *   Confluence
    *   Notion
    *   Gmail
    *   Youtube Videos
    *   GitHub
    *   Discord
    *   Airtable
    *   Google Calendar
    *   And more to come!

## Supported File Extensions

SurfSense supports a wide range of file formats, with support varying based on the ETL service configuration you choose (LlamaCloud, Unstructured, or Docling).

### Documents & Text

*   **LlamaCloud:**  `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`
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

### Audio & Video

*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured:** `.eml`, `.msg`, `.p7s`

### Browser Extension

Save webpages easily with SurfSense's cross-browser extension.
Save any webpages protected beyond authentication.

## Installation

SurfSense offers flexible installation options:

1.  **Docker Installation:** The recommended method for ease of setup, with all dependencies containerized.  See [Docker Setup Guide](DOCKER_SETUP.md)
2.  **Manual Installation:** For users who prefer a custom setup.  See [Manual Installation](https://www.surfsense.net/docs/manual-installation)

Before installation, ensure you complete the [prerequisite setup steps](https://www.surfsense.net/docs/) including PGVector, and your chosen File Processing ETL service (Unstructured.io, LlamaIndex, or Docling) with the appropriate API keys.

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

*   **FastAPI:** Modern, fast web framework for building APIs with Python.
*   **PostgreSQL with pgvector:** Database with vector search capabilities.
*   **SQLAlchemy:** SQL toolkit and ORM for database interactions.
*   **Alembic:** Database migrations tool for SQLAlchemy.
*   **FastAPI Users:** Authentication and user management (JWT & OAuth).
*   **LangGraph:** Framework for developing AI-agents.
*   **LangChain:** Framework for developing AI-powered applications.
*   **LLM Integration:** Integration with LLM models through LiteLLM.
*   **Rerankers:** Advanced result ranking for improved search relevance.
*   **Hybrid Search:** Vector similarity + full-text search with Reciprocal Rank Fusion (RRF).
*   **Vector Embeddings:** Document and text embeddings for semantic search.
*   **pgvector:** PostgreSQL extension for efficient vector similarity operations.
*   **Chonkie:** Advanced document chunking and embedding library.

### Frontend

*   **Next.js 15.2.3:** React framework with App Router, server components, code-splitting, and optimized rendering.
*   **React 19.0.0:** JavaScript library for building user interfaces.
*   **TypeScript:** Static type-checking for JavaScript.
*   **Vercel AI SDK Kit UI Stream Protocol**: To create scalable chat UI.
*   **Tailwind CSS 4.x:** Utility-first CSS framework for building custom UI designs.
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set implemented as React components.
*   **Framer Motion:** Animation library for React.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family from Vercel.
*   **React Hook Form:** Form state management and validation.
*   **Zod:** TypeScript-first schema validation with static type inference.
*   **@hookform/resolvers:** Resolvers for validation libraries with React Hook Form.
*   **@tanstack/react-table:** Headless UI for building tables & datagrids.

### DevOps

*   **Docker:** Container platform for consistent deployment.
*   **Docker Compose:** Tool for defining and running multi-container Docker applications.
*   **pgAdmin:** Web-based PostgreSQL administration tool.

### Extension
Manifest v3 on Plasmo

## Roadmap & Future Work

*   **Feature Requests & Future:** Add More Connectors, fix minor bugs, Document Podcasts
*   **Contribute:**  Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md)
*   **Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)

## Get Involved

Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to help shape the future of SurfSense and contribute your ideas and feedback!

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>

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
---