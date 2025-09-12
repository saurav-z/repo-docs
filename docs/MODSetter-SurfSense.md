# SurfSense: Your AI-Powered Research Assistant

**Unleash the power of your own personalized AI research agent that integrates with your data and favorite tools.** [Check out the original repo](https://github.com/MODSetter/SurfSense)!

---

<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">
  </a>
</div>

---
<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

---

## Key Features

*   **Personalized Knowledge Base:** Integrate your personal files and data from various sources for a customized research experience.
*   **Multi-Format File Support:** Upload documents, images, videos, and more, with support for 50+ file extensions.
*   **Advanced Search:** Quickly find information within your saved content using powerful search capabilities.
*   **AI-Powered Chat:** Interact with your content using natural language and receive cited answers.
*   **Cited Answers:** Get reliable answers with citations, just like Perplexity.
*   **Local LLM Support:** Works seamlessly with local LLMs like Ollama, ensuring privacy and control.
*   **Self-Hosted Solution:** Open-source and easy to deploy locally for maximum flexibility.
*   **Podcast Generation:**
    *   Blazingly fast podcast creation (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into audio.
    *   Support for local and multiple TTS providers (Kokoro TTS, OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs and 6000+ Embedding Models.
    *   Includes all major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes Hierarchical Indices for a 2-tiered RAG setup.
    *   Employs Hybrid Search (Semantic + Full Text Search) with Reciprocal Rank Fusion.
    *   Offers RAG as a Service API Backend.
*   **Extensive Integrations:** Connect to various external sources to expand your knowledge base.
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.
*   **Browser Extension:** Save webpages easily, even behind authentication.

---

## Supported File Extensions

SurfSense offers broad file format support, with compatibility depending on your chosen ETL service.

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

### Audio & Video

*   `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured:** `.eml`, `.msg`, `.p7s`

---

## Installation

SurfSense offers flexible installation options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest way to get SurfSense up and running, with all dependencies containerized. (Includes pgAdmin for database management via web UI, environment variable customization, flexible deployment, and detailed setup guides.)
2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For more control and customization.

**Before you install**, complete these [prerequisite setup steps](https://www.surfsense.net/docs/) including:

*   PGVector setup
*   File Processing ETL Service (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

---

## Screenshots

*   **Research Agent:**
    ![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
*   **Search Spaces:**
    ![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
*   **Manage Documents:**
    ![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)
*   **Podcast Agent:**
    ![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)
*   **Agent Chat:**
    ![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)
*   **Browser Extension:**
    ![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)
    ![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

---

## Tech Stack

### **BackEnd**

*   **FastAPI**: Modern, fast web framework for building APIs with Python
*   **PostgreSQL with pgvector**: Database with vector search capabilities for similarity searches
*   **SQLAlchemy**: SQL toolkit and ORM (Object-Relational Mapping) for database interactions
*   **Alembic**: A database migrations tool for SQLAlchemy.
*   **FastAPI Users**: Authentication and user management with JWT and OAuth support
*   **LangGraph**: Framework for developing AI-agents.
*   **LangChain**: Framework for developing AI-powered applications.
*   **LLM Integration**: Integration with LLM models through LiteLLM
*   **Rerankers**: Advanced result ranking for improved search relevance
*   **Hybrid Search**: Combines vector similarity and full-text search for optimal results using Reciprocal Rank Fusion (RRF)
*   **Vector Embeddings**: Document and text embeddings for semantic search
*   **pgvector**: PostgreSQL extension for efficient vector similarity operations
*   **Chonkie**: Advanced document chunking and embedding library
    *   Uses `AutoEmbeddings` for flexible embedding model selection
    *   `LateChunker` for optimized document chunking based on embedding model's max sequence length

---

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
Manifest v3 on Plasmo

---

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

---

## Contribute

We welcome contributions! Whether it's a ⭐, finding issues, or code improvements, every bit helps.  See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

---

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
```
Key improvements and optimizations:

*   **SEO-Optimized Hook:** A strong one-sentence opening to capture attention and highlight key value.
*   **Clear Headings:**  Organized the README using clear, descriptive headings for easy navigation.
*   **Bulleted Lists:**  Improved readability and scannability with bulleted lists for key features and supported file formats.
*   **Concise Summaries:** Summarized key features and installation steps to be more direct.
*   **Keyword Optimization:** Used relevant keywords (AI, research, knowledge base, etc.) throughout the text to improve search engine visibility.
*   **Concise Language:**  Eliminated unnecessary phrases and streamlined the language to make it easier to understand.
*   **Focus on Benefits:** Emphasized the benefits of SurfSense to users.
*   **Clear Calls to Action:** Encouraged contributions and provided direct links to the roadmap and the Discord server.
*   **Removed redundancies and combined related sections.**
*   **Included a more complete file extension listing.**
*   **Added a more complete tech stack description.**