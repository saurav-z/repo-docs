# SurfSense: Your AI-Powered Research Assistant (Open Source)

SurfSense empowers you to conduct research with unmatched efficiency by integrating your personal knowledge base with external sources, providing a powerful and customizable AI research agent. [Explore the SurfSense Repository](https://github.com/MODSetter/SurfSense).

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord)](https://discord.gg/ejRNvftDp9)

[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   **Personalized Research Hub:** Transform your data into an AI-powered knowledge base, similar to NotebookLM and Perplexity, tailored to your specific needs.
*   **Multi-Format File Support:** Upload and save content from various sources, including documents, images, videos, and more, supporting 50+ file extensions.
*   **Advanced Search Capabilities:** Quickly locate information within your saved content with powerful search functionalities.
*   **Conversational Interaction:** Engage in natural language conversations with your saved content and receive cited answers.
*   **Cited Answer Generation:** Get credible, source-cited answers similar to Perplexity.
*   **Privacy-Focused and Local LLM Support:** Works seamlessly with local LLMs, including Ollama, ensuring your data privacy.
*   **Self-Hosting Capability:** Easily deploy and manage your research agent locally, thanks to its open-source nature.
*   **AI-Powered Podcasts:** 
    *   Generate podcasts rapidly (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into audio content.
    *   Supports local and external TTS providers (Kokoro TTS, OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs
    *   Supports 6000+ Embedding Models
    *   Supports all major Rerankers
    *   Utilizes Hierarchical Indices (2 tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **External Source Integration:** Connects with various sources to expand your research scope:
    *   Search Engines (Tavily, LinkUp)
    *   Slack
    *   Linear
    *   Jira
    *   ClickUp
    *   Confluence
    *   Notion
    *   Gmail
    *   YouTube
    *   GitHub
    *   Discord
    *   Airtable
    *   Google Calendar
    *   ...and more.
*   **Cross-Browser Extension:** Save any webpage you like with the SurfSense extension.

## Supported File Extensions

SurfSense leverages multiple ETL services for comprehensive file format support. Please note that file support depends on your chosen ETL service configuration (LlamaCloud, Unstructured, or Docling).

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

## Getting Started

SurfSense offers two installation options:

1.  **Docker Installation:** The simplest method, leveraging Docker for a containerized environment.
2.  **Manual Installation:** For users preferring more control or custom configurations.

Detailed setup instructions are available in the [Installation Guides](https://www.surfsense.net/docs/).

## Screenshots

**(Include your screenshots here, with appropriate captions for SEO.)**

*   **Research Agent**  - *(Caption)*

    ![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)

*   **Search Spaces**  - *(Caption)*

    ![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)

*   **Manage Documents** - * (Caption)*

    ![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)

*   **Podcast Agent**  - * (Caption)*

    ![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)

*   **Agent Chat** - *(Caption)*

    ![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)

*   **Browser Extension** - * (Caption)*

    ![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)
    ![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### Backend

*   FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LiteLLM, Advanced RAG Techniques, Hybrid Search, Vector Embeddings, pgvector, Chonkie, `AutoEmbeddings`

### Frontend

*   Next.js 15.2.3, React 19.0.0, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS 4.x, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table

### DevOps

*   Docker, Docker Compose, pgAdmin

### Extension

*   Manifest v3 on Plasmo

## Future Work

*   Expanding connector support.
*   Ongoing bug fixes.
*   Podcast Documentation
*   Enhanced podcast features

## Contribute

We warmly welcome contributions!  Check out our [CONTRIBUTING.md](CONTRIBUTING.md) file for details.

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
---