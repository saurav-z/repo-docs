![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

# SurfSense: Your Customizable AI Research Agent

**SurfSense empowers you to research, synthesize, and extract knowledge from your digital world by connecting your personal knowledge base to multiple external sources.** ([Back to the GitHub Repository](https://github.com/MODSetter/SurfSense))

[![Discord](https://img.shields.io/discord/1359368468260192417?logo=discord&label=Discord)](https://discord.gg/ejRNvftDp9)

[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   **Personalized Knowledge Base:** Seamlessly integrate your personal files, documents, and data into your research workflow.
*   **Multiple File Format Support:**  Upload a wide variety of file types, including documents, images, videos, and more (**50+ file extensions supported**).
*   **Powerful Search:** Quickly locate information within your saved content.
*   **AI-Powered Chat:** Interact with your data using natural language and receive cited answers, similar to tools like Perplexity.
*   **Cited Answers:**  Get direct source citations for enhanced reliability.
*   **Local LLM Support & Privacy:** Works flawlessly with local LLMs, ensuring privacy and control.
*   **Self-Hosted:** Open-source and easy to deploy locally.
*   **Blazing Fast Podcast Agent:**  Generate podcasts from conversations with support for multiple TTS providers (OpenAI, Azure, Google Vertex AI & local TTS providers).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecone, Cohere, Flashrank, etc.)
    *   Utilizes Hierarchical Indices (2 tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **External Source Integration:** Connects to:
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
    *   Airtable
    *   Google Calendar
    *   And more to come...
*   **Cross-Browser Extension:**  Save webpages, even those behind authentication.

## Supported File Extensions

File format support varies based on your chosen ETL service. Choose from Unstructured.io, LlamaCloud, or Docling based on your needs.

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

## Installation

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest way to get SurfSense up and running with all dependencies containerized.
2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)** - For users who prefer more control over their setup.

Both methods provide detailed instructions.  Before installation, ensure you complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:

*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute

Contributions are highly encouraged!  Star the repository, report issues, or submit pull requests.  See our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Roadmap

Stay updated on our development progress.
**View the Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>
<br>
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