<div align="center">
  <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
</div>

# SurfSense: Your AI-Powered Research Assistant for Personalized Knowledge Discovery

SurfSense empowers you to conduct research like never before by connecting to your personal knowledge base and external sources, giving you a powerful, customizable AI research agent.  [Explore the original repository](https://github.com/MODSetter/SurfSense).

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **Intelligent Research Hub:** Create a personalized AI research environment, similar to NotebookLM and Perplexity, but connected to your data.
*   📁 **Comprehensive File Support:** Upload and save content from various file formats, including documents, images, and videos with support for **50+ file extensions**.
*   🔍 **Advanced Search Capabilities:** Quickly and efficiently search your saved content to find exactly what you need.
*   💬 **Conversational Interaction:** Engage in natural language conversations with your saved content and receive cited answers.
*   📄 **Cited Answers:** Get reliable answers with proper citations, just like Perplexity.
*   🔔 **Privacy-Focused and Local LLM Support:** Works seamlessly with local LLMs like Ollama for enhanced privacy.
*   🏠 **Self-Hostable:**  Open source and easy to deploy on your local machine.
*   🎙️ **AI-Powered Podcasts:**
    *   Quickly generate podcasts from your conversations.
    *   Convert chat conversations to audio.
    *   Support for local and multiple TTS providers (Kokoro TTS, OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes Hierarchical Indices (2-tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **Extensive External Data Integrations:** Connect to a wide range of sources:
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
*   🔖 **Web Clipper Browser Extension:**  Save any webpage you like with the SurfSense extension.

## Supported File Extensions

**Note:** File format support depends on your ETL service configuration. LlamaCloud supports 50+ formats, Unstructured supports 34+ core formats, and Docling (core formats, local processing, privacy-focused, no API key).

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

### Audio & Video *(Always Supported)*
*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication
*   **Unstructured:** `.eml`, `.msg`, `.p7s`

##  Get Started

SurfSense offers two installation options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: The simplest way to get SurfSense running.  Includes pgAdmin for database management, supports environment variable customization via `.env` file, and flexible deployment options.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) for detailed instructions
    *   For deployment scenarios and options, see [Deployment Guide](DEPLOYMENT_GUIDE.md)

2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)**: For users who prefer greater control or need custom configurations.

Before installing, complete the [prerequisite setup steps](https://www.surfsense.net/docs/) including:
- PGVector setup
- **File Processing ETL Service** (choose one):
  - Unstructured.io API key (supports 34+ formats)
  - LlamaIndex API key (enhanced parsing, supports 50+ formats)
  - Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
- Other required API keys

## Screenshots

<div align="center">
  <img src="https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4" alt="Research Agent" width="700">
  <img src="https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099" alt="Search Spaces" width="700">
  <img src="https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d" alt="Manage Documents" width="700">
  <img src="https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c" alt="Podcast Agent" width="700">
  <img src="https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491" alt="Agent Chat" width="700">
  <img src="https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40" alt="Browser Extension 1" width="350">
  <img src="https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7" alt="Browser Extension 2" width="350">
</div>

## Tech Stack

### **Backend**

*   **FastAPI**: Web framework
*   **PostgreSQL with pgvector**: Database with vector search
*   **SQLAlchemy**:  SQL toolkit and ORM
*   **Alembic**:  Database migrations
*   **FastAPI Users**: Authentication and user management
*   **LangGraph**: Framework for developing AI-agents.
*   **LangChain**: Framework for developing AI-powered applications.
*   **LLM Integration**: Integration with LLM models through LiteLLM
*   **Rerankers**: Advanced result ranking
*   **Hybrid Search**: Combines vector similarity and full-text search
*   **Vector Embeddings**: Document and text embeddings
*   **pgvector**: PostgreSQL extension for vector similarity
*   **Chonkie**: Document chunking and embedding library
    *   Uses `AutoEmbeddings` for flexible embedding model selection
    *   `LateChunker` for optimized document chunking

### **Frontend**

*   **Next.js 15.2.3**: React framework
*   **React 19.0.0**: UI library
*   **TypeScript**:  Static type-checking
*   **Vercel AI SDK Kit UI Stream Protocol**: Scalable chat UI
*   **Tailwind CSS 4.x**: Utility-first CSS framework
*   **Shadcn**: Headless components library.
*   **Lucide React**: Icon set
*   **Framer Motion**: Animation library
*   **Sonner**: Toast notification library
*   **Geist**: Font family from Vercel
*   **React Hook Form**: Form state management
*   **Zod**: TypeScript-first schema validation
*   **@hookform/resolvers**: Validation libraries with React Hook Form
*   **@tanstack/react-table**: Headless UI for tables & datagrids

### **DevOps**

*   **Docker**: Container platform
*   **Docker Compose**: Multi-container application tool
*   **pgAdmin**: Web-based PostgreSQL administration tool

### **Extension**

*   Manifest v3 on Plasmo

## Future Development

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute

Your contributions are welcome!  From a simple star to fixing bugs, all contributions are valuable. See our [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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