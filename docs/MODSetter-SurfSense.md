<div align="center">
    <a href="https://github.com/MODSetter/SurfSense">
        <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
    </a>
</div>

<div align="center">
    <a href="https://discord.gg/ejRNvftDp9">
        <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord" alt="Discord">
    </a>
</div>

# SurfSense: Your Customizable AI Research Agent

**SurfSense is a powerful, open-source AI research agent that lets you integrate your personal knowledge base with external sources for in-depth research and discovery.**

[Explore the SurfSense Repository](https://github.com/MODSetter/SurfSense)

## Key Features

*   **Customizable AI Research:** Build your own private NotebookLM and Perplexity-like experience tailored to your needs.
*   **Multiple File Format Support:** Upload and analyze content from your personal files, supporting **50+ file extensions**.
*   **Powerful Search:** Quickly find anything within your saved content.
*   **Natural Language Chat:** Interact with your saved content using natural language and receive cited answers.
*   **Cited Answers:** Get reliable, source-cited answers just like Perplexity.
*   **Privacy & Local LLM Support:** Works seamlessly with local LLMs (like Ollama) for enhanced privacy.
*   **Self-Hostable:** Deploy SurfSense locally with ease; open-source and ready to use.
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs
    *   Supports 6000+ Embedding Models
    *   Supports major Rerankers (Pinecode, Cohere, Flashrank, etc.)
    *   Utilizes Hierarchical Indices (2-tiered RAG setup)
    *   Employs Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion)
    *   RAG as a Service API Backend.
*   **Podcast Generation Agent:**
    *   Blazingly fast podcast generation (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into audio content.
    *   Supports local TTS providers (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   **External Source Integration:** Connect to a wide range of sources:
    *   Search Engines (Tavily, LinkUp)
    *   Slack
    *   Linear
    *   Jira
    *   ClickUp
    *   Confluence
    *   Gmail
    *   Notion
    *   YouTube Videos
    *   GitHub
    *   Discord
    *   Airtable
    *   Google Calendar
    *   And more!
*   **Cross-Browser Extension:** Save any webpage you like with the SurfSense extension.

## Supported File Extensions

**Note:** File format support depends on your ETL service configuration (LlamaCloud, Unstructured, Docling).

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

`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication

*   **Unstructured:** `.eml`, `.msg`, `.p7s`

## Installation

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** (Recommended): The easiest way to get started, with all dependencies containerized. Includes pgAdmin for database management and supports environment variable customization via `.env` files. See the [Docker Setup Guide](DOCKER_SETUP.md) for detailed instructions.
2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)**: For users who prefer more control or need custom configurations.

Before you start, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including PGVector and an ETL service (Unstructured.io, LlamaIndex, or Docling).

## Screenshots

<details>
<summary>Click to Expand</summary>

**Research Agent**
<img src="https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4" alt="Research Agent" width="70%">

**Search Spaces**
<img src="https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099" alt="Search Spaces" width="70%">

**Manage Documents**
<img src="https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d" alt="Manage Documents" width="70%">

**Podcast Agent**
<img src="https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c" alt="Podcast Agent" width="70%">

**Agent Chat**
<img src="https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491" alt="Agent Chat" width="70%">

**Browser Extension**
<img src="https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40" alt="Extension 1" width="70%">
<img src="https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7" alt="Extension 2" width="70%">
</details>

## Tech Stack

### BackEnd

*   **FastAPI:** Web framework for building APIs (Python).
*   **PostgreSQL with pgvector:** Database with vector search capabilities.
*   **SQLAlchemy:** SQL toolkit and ORM for database interactions.
*   **Alembic:** Database migrations tool.
*   **FastAPI Users:** Authentication and user management.
*   **LangGraph:** Framework for developing AI-agents.
*   **LangChain:** Framework for developing AI-powered applications.
*   **LLM Integration:** LLM models via LiteLLM.
*   **Rerankers:** Advanced result ranking for improved search relevance.
*   **Hybrid Search:** Semantic + Full Text Search with Reciprocal Rank Fusion (RRF).
*   **Vector Embeddings:** Document and text embeddings for semantic search.
*   **pgvector:** PostgreSQL extension for efficient vector similarity.
*   **Chonkie:** Advanced document chunking and embedding library with `AutoEmbeddings` for model selection and `LateChunker` for optimized chunking.

### FrontEnd

*   **Next.js 15.2.3:** React framework with App Router, server components, and optimized rendering.
*   **React 19.0.0:** JavaScript library for building user interfaces.
*   **TypeScript:** Static type-checking for JavaScript.
*   **Vercel AI SDK Kit UI Stream Protocol:** Scalable chat UI.
*   **Tailwind CSS 4.x:** Utility-first CSS framework.
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set.
*   **Framer Motion:** Animation library.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family from Vercel.
*   **React Hook Form:** Form state management and validation.
*   **Zod:** TypeScript-first schema validation.
*   **@hookform/resolvers:** Validation library resolvers.
*   **@tanstack/react-table:** Headless UI for tables & datagrids.

### DevOps

*   **Docker:** Container platform for consistent deployment.
*   **Docker Compose:** Tool for multi-container applications.
*   **pgAdmin:** Web-based PostgreSQL administration.

### Extension

*   Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Fix minor bugs.
*   Improve Podcast Documentation

## Contribute

Contributions are highly encouraged!  From ⭐ to issue creation or code contributions, all help is welcome.  See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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
Key improvements and SEO considerations:

*   **Clear, Concise Hook:** "SurfSense is a powerful, open-source AI research agent that lets you integrate your personal knowledge base with external sources for in-depth research and discovery."  This immediately explains the core benefit.
*   **Keyword Optimization:**  Includes relevant keywords throughout: "AI research agent," "knowledge base," "search," "customizable," "self-hostable," "LLM," "Podcast."
*   **Detailed Feature Breakdown:**  Uses bullet points for easy readability and emphasizes key features.
*   **Headings & Subheadings:**  Organizes information logically for improved SEO and user experience.  Uses `h1` and `h2` tags, which search engines prioritize.
*   **Call to Action (CTA):** Encourages users to "Explore the SurfSense Repository."
*   **Image Alt Text:** Added `alt` text to all images for accessibility and SEO.
*   **Clear Installation Instructions:**  Highlights Docker and Manual installation, with links to the appropriate documentation.
*   **Emphasis on Benefits:**  Focuses on what users *gain* (e.g., "Customizable AI Research," "Natural Language Chat").
*   **Tech Stack Section:** Clearly lists the technologies used, potentially attracting users interested in those technologies.
*   **Contribution Guidance:** Makes it easy for users to know how to contribute and get involved.
*   **Star History:** Added the Star History chart for credibility.
*   **Discord Badge and Link:** Provides a clear way to connect with the community.
*   **Links back to the original repo:** Included.