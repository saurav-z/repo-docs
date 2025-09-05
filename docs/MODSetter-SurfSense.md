![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

# SurfSense: Your AI-Powered Research Assistant

**SurfSense empowers you to conduct in-depth research by connecting your personal knowledge base with external sources, offering a customizable and private research experience.** ([Back to Original Repo](https://github.com/MODSetter/SurfSense))

<div align="center">
<a href="https://discord.gg/ejRNvftDp9">
<img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">
</a>
</div>

<div align="center">
<a href="https://trendshift.io/repositories/13606" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

## Key Features

*   💡 **Personalized Research Powerhouse:** Create your own customizable AI research agent, similar to NotebookLM and Perplexity, but tailored to your needs.
*   📁 **Extensive File Format Support:** Upload and save content from diverse file formats, including documents, images, videos, and more (50+ file extensions supported).
*   🔍 **Advanced Search Capabilities:** Quickly search and find specific information within your saved content.
*   💬 **Natural Language Interaction:** Chat with your saved content and receive cited answers for in-depth understanding.
*   📄 **Cited Answers:** Get answers with citations, just like Perplexity.
*   🔔 **Privacy & Local LLM Support:** Works seamlessly with local LLMs like Ollama, ensuring privacy.
*   🏠 **Self-Hosted & Open Source:** Deploy SurfSense locally with ease.
*   🎙️ **Podcast Generation:**
    *   Blazingly fast podcast creation (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into engaging audio content.
    *   Supports local TTS providers (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes Hierarchical Indices (2-tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **Extensive External Source Integration:**
    *   Search Engines: Tavily, LinkUp.
    *   Collaboration Tools: Slack, Linear, Jira, ClickUp, Confluence, Notion.
    *   Communication & Content: Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.

## Supported File Extensions

> **Note:** File format support depends on your ETL service configuration (LlamaCloud, Unstructured, or Docling).

### Documents & Text

*   **LlamaCloud:** .pdf, .doc, .docx, .docm, .dot, .dotm, .rtf, .txt, .xml, .epub, .odt, .wpd, .pages, .key, .numbers, .602, .abw, .cgm, .cwk, .hwp, .lwp, .mw, .mcw, .pbd, .sda, .sdd, .sdp, .sdw, .sgl, .sti, .sxi, .sxw, .stw, .sxg, .uof, .uop, .uot, .vor, .wps, .zabw
*   **Unstructured:** .doc, .docx, .odt, .rtf, .pdf, .xml, .txt, .md, .markdown, .rst, .html, .org, .epub
*   **Docling:** .pdf, .docx, .html, .htm, .xhtml, .adoc, .asciidoc

### Presentations

*   **LlamaCloud:** .ppt, .pptx, .pptm, .pot, .potm, .potx, .odp, .key
*   **Unstructured:** .ppt, .pptx
*   **Docling:** .pptx

### Spreadsheets & Data

*   **LlamaCloud:** .xlsx, .xls, .xlsm, .xlsb, .xlw, .csv, .tsv, .ods, .fods, .numbers, .dbf, .123, .dif, .sylk, .slk, .prn, .et, .uos1, .uos2, .wk1, .wk2, .wk3, .wk4, .wks, .wq1, .wq2, .wb1, .wb2, .wb3, .qpw, .xlr, .eth
*   **Unstructured:** .xls, .xlsx, .csv, .tsv
*   **Docling:** .xlsx, .csv

### Images

*   **LlamaCloud:** .jpg, .jpeg, .png, .gif, .bmp, .svg, .tiff, .webp, .html, .htm, .web
*   **Unstructured:** .jpg, .jpeg, .png, .bmp, .tiff, .heic
*   **Docling:** .jpg, .jpeg, .png, .bmp, .tiff, .tif, .webp

### Audio & Video (Always Supported)

*   .mp3, .mpga, .m4a, .wav, .mp4, .mpeg, .webm

### Email & Communication

*   **Unstructured:** .eml, .msg, .p7s

### 🔖 Cross-Browser Extension

*   Save any webpage with the SurfSense browser extension.
*   Save any webpages protected beyond authentication.

## 🚀 Roadmap & Future Development

**SurfSense is under active development.** Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) and contribute to its future.

**Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)

## Getting Started

### Installation Options

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation):** Easiest setup with all dependencies containerized.
    *   Includes pgAdmin for database management.
    *   Customizable via .env file.
    *   Flexible deployment options.
    *   See [Docker Setup Guide](DOCKER_SETUP.md).
    *   See [Deployment Guide](DEPLOYMENT_GUIDE.md).

2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation):** For users with more control.

**Prerequisites:**

*   PGVector setup
*   File Processing ETL Service (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

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

*   FastAPI
*   PostgreSQL with pgvector
*   SQLAlchemy
*   Alembic
*   FastAPI Users
*   LangGraph
*   LangChain
*   LLM Integration (LiteLLM)
*   Rerankers
*   Hybrid Search
*   Vector Embeddings
*   pgvector
*   Chonkie

### Frontend

*   Next.js 15.2.3
*   React 19.0.0
*   TypeScript
*   Vercel AI SDK Kit UI Stream Protocol
*   Tailwind CSS 4.x
*   Shadcn
*   Lucide React
*   Framer Motion
*   Sonner
*   Geist
*   React Hook Form
*   Zod
*   @hookform/resolvers
*   @tanstack/react-table

### DevOps

*   Docker
*   Docker Compose
*   pgAdmin

### Extension

*   Manifest v3 on Plasmo

## Future Work
*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute

We welcome contributions of all kinds! Whether it's a star, an issue report, or code changes, your help is appreciated.
For detailed contribution guidelines, please see our [CONTRIBUTING.md](CONTRIBUTING.md) file.

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
```
Key improvements and SEO considerations:

*   **Clear and Concise Hook:** The one-sentence hook at the beginning immediately grabs attention.
*   **Keyword Optimization:** The title and headings use relevant keywords (e.g., "AI research agent," "personal knowledge base," "self-hosted").
*   **Structured Headings:**  Organizes the content for readability and SEO benefits.  Uses H2 and H3 tags appropriately.
*   **Bulleted Lists:** Makes key features easy to scan and digest.
*   **Concise Language:** The text is more direct and avoids unnecessary jargon.
*   **Emphasis on Benefits:** Highlights the advantages of using SurfSense.
*   **Internal and External Linking:** Includes links to relevant resources, including the original repo, roadmap, and documentation.  Also links to external resources where appropriate.
*   **Alt Text for Images:**  Provides descriptions for images, which is crucial for accessibility and SEO.
*   **Clear Call to Action:** Encourages users to contribute and join the Discord.
*   **Tech Stack:** More organized and descriptive.
*   **Star History:** Include Star History to show project growth and interest.
*   **Removed redundant images**.
*   **Formatted as Markdown:**  Easily readable and optimized for GitHub and other platforms.

This improved README is more informative, user-friendly, and search engine optimized, making it easier for people to find and understand SurfSense.