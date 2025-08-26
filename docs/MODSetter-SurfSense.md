<!-- Improved & Summarized README -->
<!-- SurfSense: Your customizable AI research assistant. -->

<div align="center">
  <a href="https://github.com/MODSetter/SurfSense">
    <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
  </a>
</div>

<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord&style=social" alt="Discord">
  </a>
</div>

# SurfSense: Your Personalized AI Research Powerhouse

SurfSense transforms how you research by integrating your personal knowledge base with AI, providing a highly customizable and powerful research agent that connects to your data and external sources.

[**View the original repository on GitHub**](https://github.com/MODSetter/SurfSense)

<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## Key Features

*   **Personalized Research:**  Build your own private, customizable research agent, similar to NotebookLM and Perplexity, tailored to your needs.
*   **Versatile Data Integration:** Connect to a wide range of sources, including:
    *   **External Sources:** Search Engines (Tavily, LinkUp), Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Google Calendar, and more.
    *   **File Uploads:** Supports documents, images, videos, and over 50 file extensions.
*   **Powerful Search & Retrieval:** Quickly search and find information within your saved content.
*   **AI-Powered Interaction:**  Chat with your saved content in natural language and receive cited answers.
*   **Privacy-Focused:**  Works seamlessly with local LLMs like Ollama, ensuring data privacy.
*   **Self-Hosting:** Open source and easy to deploy locally.
*   **Podcast Generation:** Quickly create engaging audio content from your chats or documents.
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecode, Cohere, Flashrank etc)
    *   Hierarchical Indices (2 tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **Cross-Browser Extension:** Save webpages easily using the SurfSense extension.

## Supported File Extensions

*(File format support depends on your ETL service configuration.)*

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

## Installation

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest method, with all dependencies containerized, including pgAdmin for database management.  Uses environment variables for easy configuration.
2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For users preferring more control and customization.

**Before Installation:** Complete prerequisite setup steps, including PGVector, a File Processing ETL Service (Unstructured.io or LlamaIndex API Key or Docling), and other required API keys.  See [Installation Guides](https://www.surfsense.net/docs/) for details.

## Screenshots

**(Images showing the Research Agent, Search Spaces, Document Management, Podcast Agent, Agent Chat, and Browser Extension would go here.  Use the provided image links from the original README.)**

## Tech Stack

### Backend

*   **FastAPI:** Web framework.
*   **PostgreSQL with pgvector:** Database with vector search.
*   **SQLAlchemy:** ORM for database interactions.
*   **Alembic:** Database migrations.
*   **FastAPI Users:** Authentication and user management.
*   **LangGraph & LangChain:** Frameworks for AI agent development.
*   **LLM Integration:** Through LiteLLM.
*   **Rerankers:** Result ranking.
*   **Hybrid Search:** Semantic + Full-text search.
*   **Vector Embeddings:** for semantic search
*   **pgvector:** PostgreSQL extension for vector similarity operations
*   **Chonkie:** Document chunking library
 - Uses `AutoEmbeddings` for flexible embedding model selection
 -  `LateChunker` for optimized document chunking based on embedding model's max sequence length

### Frontend

*   **Next.js 15.2.3:** React framework
*   **React 19.0.0:** UI library.
*   **TypeScript:** Static type-checking.
*   **Vercel AI SDK Kit UI Stream Protocol**
*   **Tailwind CSS 4.x:** CSS framework.
*   **Shadcn:** Headless component library.
*   **Lucide React:** Icon set.
*   **Framer Motion:** Animation library.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family.
*   **React Hook Form:** Form state management.
*   **Zod:** Schema validation.
*   **@hookform/resolvers:** Validation library integration.
*   **@tanstack/react-table:** Table/datagrid UI.

### DevOps

*   **Docker:** Container platform.
*   **Docker Compose:** Multi-container application tool.
*   **pgAdmin:** PostgreSQL administration tool.

### Extension

* Manifest v3 on Plasmo

## Roadmap & Community

**SurfSense is actively evolving!**

*   **Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   **Contribute:** Contributions are welcomed!
*   **Discord:** [Join the SurfSense Discord](https://discord.gg/ejRNvftDp9) to provide feedback and shape the future.

## Contribute

We welcome contributions! Please see our [CONTRIBUTING.md](CONTRIBUTING.md) file for details on how to contribute.