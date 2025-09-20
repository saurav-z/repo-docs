# SurfSense: Your AI-Powered Research Assistant

**SurfSense transforms your research workflow by connecting your personal knowledge base to external sources, offering a customizable and private AI research experience.**  [View the original repo on GitHub](https://github.com/MODSetter/SurfSense)

[<img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">](https://discord.gg/ejRNvftDp9)

[<img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>](https://trendshift.io/repositories/13606)

[Video](https://github.com/user-attachments/assets/d9221908-e0de-4b2f-ac3a-691cf4b202da)

[Podcast Sample](https://github.com/user-attachments/assets/a0a16566-6967-4374-ac51-9b3e07fbecd7)

## Key Features:

*   💡 **Personalized AI Research:** Create a private NotebookLM and Perplexity experience, tailored to your knowledge base.
*   📁 **Extensive File Format Support:** Upload documents, images, videos, and more with support for 50+ file extensions.
*   🔍 **Advanced Search:** Quickly find information within your saved content.
*   💬 **Natural Language Interaction:** Chat with your content and receive cited answers.
*   📄 **Cited Answers:** Get reliable, source-cited answers, similar to Perplexity.
*   🔔 **Privacy-Focused & Local LLM Support:** Works seamlessly with local LLMs like Ollama.
*   🏠 **Self-Hostable:** Open-source and easy to deploy locally.
*   🎙️ **Blazing-Fast Podcast Generation:**
    *   Generates 3-minute podcasts in under 20 seconds.
    *   Converts chat conversations into audio content.
    *   Supports local TTS providers (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs and 6000+ embedding models.
    *   Integrates with major rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Employs Hierarchical Indices (2-tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search) with Reciprocal Rank Fusion.
    *   RAG as a Service API Backend.
*   ℹ️ **Connect to External Sources:**
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more to come.
*   🌐 **Cross-Browser Extension:** Save webpages directly to your knowledge base.

## Supported File Extensions

**Note:** File format support depends on your ETL service configuration (LlamaCloud, Unstructured, or Docling).

### [Documents & Text](https://www.surfsense.net/docs/file-formats)
**LlamaCloud**: `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`

**Unstructured**: `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub`

**Docling**: `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc`

### [Presentations](https://www.surfsense.net/docs/file-formats)
**LlamaCloud**: `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key`

**Unstructured**: `.ppt`, `.pptx`

**Docling**: `.pptx`

### [Spreadsheets & Data](https://www.surfsense.net/docs/file-formats)
**LlamaCloud**: `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, `.123`, `.dif`, `.sylk`, `.slk`, `.prn`, `.et`, `.uos1`, `.uos2`, `.wk1`, `.wk2`, `.wk3`, `.wk4`, `.wks`, `.wq1`, `.wq2`, `.wb1`, `.wb2`, `.wb3`, `.qpw`, `.xlr`, `.eth`

**Unstructured**: `.xls`, `.xlsx`, `.csv`, `.tsv`

**Docling**: `.xlsx`, `.csv`

### [Images](https://www.surfsense.net/docs/file-formats)
**LlamaCloud**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web`

**Unstructured**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`

**Docling**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

### [Audio & Video](https://www.surfsense.net/docs/file-formats) *(Always Supported)*
`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### [Email & Communication](https://www.surfsense.net/docs/file-formats)
**Unstructured**: `.eml`, `.msg`, `.p7s`

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

## Tech Stack:

*   **Backend:** FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LLM Integration (LiteLLM), Rerankers, Hybrid Search, Vector Embeddings, pgvector, Chonkie
*   **Frontend:** Next.js, React, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table
*   **DevOps:** Docker, Docker Compose, pgAdmin
*   **Extension:** Manifest v3 on Plasmo

## Installation:

SurfSense offers two installation methods:

1.  [Docker Installation](https://www.surfsense.net/docs/docker-installation)
2.  [Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)

Before installing, complete the [prerequisite setup steps](https://www.surfsense.net/docs/).

## Contribute & Get Involved:

SurfSense is actively evolving, and contributions are welcome! [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

**Join the Community:**  [SurfSense Discord](https://discord.gg/ejRNvftDp9) to contribute and shape the future!

## Roadmap:

Stay updated on development progress via the [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2).

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