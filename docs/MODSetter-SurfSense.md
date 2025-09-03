[![Discord](https://img.shields.io/discord/1359368468260192417?logo=discord)](https://discord.gg/ejRNvftDp9)

<div align="center">
  <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
</div>

# SurfSense: Your Customizable AI Research Assistant 🚀

**SurfSense empowers you to conduct research, analyze information, and generate content from your personal knowledge base and diverse external sources, all with the power of AI.** ([Original Repo](https://github.com/MODSetter/SurfSense))

<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank" rel="noopener noreferrer">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="SurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## Key Features

*   **Personalized AI Research:** Create your own highly customizable AI research assistant, similar to NotebookLM and Perplexity, but integrated with your personal knowledge.
*   **Multi-Source Integration:** Connect to and search across a wide variety of external sources including:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Gmail, Notion, YouTube, GitHub, Discord, Google Calendar, and more!
*   **Extensive File Format Support:** Upload and save content from various file types, including documents, images, videos, and more. (Supports 50+ file extensions via LlamaCloud, and other options.)
*   **Powerful Search Capabilities:** Quickly find information within your saved content using advanced search techniques.
*   **AI-Powered Chat:** Interact with your saved content using natural language and receive cited answers.
*   **Privacy-Focused & Local LLM Support:** Works seamlessly with local LLMs like Ollama, ensuring data privacy.
*   **Self-Hosting:**  Open-source and designed for easy local deployment.
*   **Podcast Generation:**
    *   Blazingly fast podcast creation (3-minute podcast in under 20 seconds).
    *   Convert conversations into engaging audio content.
    *   Support for local and multiple TTS providers (Kokoro TTS, OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs and 6000+ Embedding Models.
    *   Integrates with all major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Employs Hierarchical Indices (2-tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **Cross-Browser Extension:** Save webpages easily, even those behind authentication.

## Supported File Extensions

SurfSense supports a broad range of file formats. Please note that the specific file format support depends on the ETL service you configure.  (LlamaCloud, Unstructured, and Docling are options with different capabilities)

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

### Audio & Video

*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured**: `.eml`, `.msg`, `.p7s`

### 🔖 Cross Browser Extension

*   The SurfSense extension can be used to save any webpage you like.
*   Its main usecase is to save any webpages protected beyond authentication.

---
---
<p align="center">
  <a href="https://handbook.opencoreventures.com/catalyst-sponsorship-program/" target="_blank" rel="noopener noreferrer">
    <img 
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4" 
      alt="Catalyst Sponsorship Program" 
      width="600"
    />
  </a>
</p>

---
---

## Get Involved & Contribute

**SurfSense is under active development!** Your contributions are welcome to help shape its future.

*   **Join the Community:**  Connect with other users and developers on the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to share ideas and provide feedback.
*   **Roadmap:** Stay updated on the development progress and upcoming features via our public roadmap: [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   **Contribute:**  Contribute by creating issues, submitting code, or giving a ⭐ on the repo. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## Installation

SurfSense offers two installation methods:

1.  **Docker Installation:** ([Docker Installation Guide](https://www.surfsense.net/docs/docker-installation)) The easiest way to get SurfSense up and running, with all dependencies containerized.
    *   Includes pgAdmin for database management.
    *   Customization via `.env` files.
    *   Flexible deployment options.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md) for details.

2.  **Manual Installation:** ([Manual Installation Guide](https://www.surfsense.net/docs/manual-installation)) For users who prefer greater control over their setup.

**Prerequisites:**
Before installing, ensure you have completed the prerequisite setup steps:
*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Screenshots

### Research Agent

![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)

### Search Spaces

![search_spaces](https://github.com/user-attachments/assets/e254c386-f937-44b6-9e9d-770db583d099)

### Manage Documents

![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)

### Podcast Agent

![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)

### Agent Chat

![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)

### Browser Extension

![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)

![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### **Backend**

*   **FastAPI:** Web framework for building APIs with Python
*   **PostgreSQL with pgvector:** Database with vector search capabilities
*   **SQLAlchemy:** SQL toolkit and ORM
*   **Alembic:** Database migrations tool
*   **FastAPI Users:** Authentication and user management
*   **LangGraph:** Framework for developing AI-agents.
*   **LangChain:** Framework for developing AI-powered applications.
*   **LLM Integration:** Integration with LLM models through LiteLLM
*   **Rerankers:** Advanced result ranking
*   **Hybrid Search:** Combines vector similarity and full-text search
*   **Vector Embeddings:** Document and text embeddings
*   **pgvector:** PostgreSQL extension for vector similarity operations
*   **Chonkie:** Document chunking and embedding library
    *   Uses `AutoEmbeddings` for flexible embedding model selection
    *   `LateChunker` for optimized document chunking

### **Frontend**

*   **Next.js 15.2.3:** React framework
*   **React 19.0.0:** JavaScript library
*   **TypeScript:** Static type-checking for JavaScript
*   **Vercel AI SDK Kit UI Stream Protocol:** Create scalable chat UI.
*   **Tailwind CSS 4.x:** CSS framework
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set
*   **Framer Motion:** Animation library
*   **Sonner:** Toast notification library
*   **Geist:** Font family from Vercel
*   **React Hook Form:** Form state management
*   **Zod:** TypeScript-first schema validation
*   **@hookform/resolvers:** Resolvers for validation libraries
*   **@tanstack/react-table:** Headless UI for tables & datagrids

### **DevOps**

*   **Docker:** Container platform
*   **Docker Compose:** Tool for running multi-container Docker applications
*   **pgAdmin:** Web-based PostgreSQL administration tool

### **Extension**

*   Manifest v3 on Plasmo

## Future Development

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>