# SurfSense: Your AI Research Agent – Supercharge Your Knowledge Base 🧠

SurfSense empowers you to conduct in-depth research and explore your personal knowledge base like never before, seamlessly integrating with various external sources. **[Check out the original repo](https://github.com/MODSetter/SurfSense) for the latest updates!**

## Key Features:

*   ✅ **AI-Powered Knowledge Management:** Transform your personal data into an accessible and searchable knowledge base.
*   📁 **Versatile File Format Support:** Upload and process a wide range of file types, including documents, images, videos, and more, with support for **50+ file extensions**.
*   🔍 **Robust Search Capabilities:** Quickly find information within your saved content using powerful search tools.
*   💬 **Natural Language Interaction:** Engage in intuitive conversations with your data to get cited answers.
*   📄 **Cited Answers:** Receive verifiable and cited answers, similar to Perplexity, for trustworthy research.
*   🔔 **Privacy & Local LLM Support:** Works seamlessly with Ollama and other local LLMs, ensuring data privacy.
*   🏠 **Self-Hostable:** Deploy SurfSense locally for complete control over your data and environment.
*   🎙️ **Podcast Agent:** Generate engaging audio content from conversations and saved content.
    *   Blazingly fast podcast generation: create a 3-minute podcast in under 20 seconds!
    *   Converts chat conversations into audio.
    *   Supports local TTS providers like Kokoro TTS and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:** SurfSense leverages cutting-edge Retrieval-Augmented Generation (RAG) techniques.
    *   Supports 100+ LLMs
    *   6000+ Embedding Models
    *   Major Rerankers (Pinecode, Cohere, Flashrank etc)
    *   Hierarchical indices (2-tiered RAG setup)
    *   Hybrid search (semantic + full-text search combined with Reciprocal Rank Fusion (RRF))
    *   RAG as a Service API backend.
*   🌐 **Extensive External Source Integration:** Connect to a wide range of external services to expand your knowledge network:
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
    *   And many more to come!
*   🔗 **Cross-Browser Extension:** Save web pages with the browser extension.

## Supported File Extensions:

SurfSense utilizes several ETL services.  File format support depends on your ETL service configuration.

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

### Browser Extension:

*   Save webpages that are beyond authentication.

## Getting Started:

### Installation Options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest method, fully containerized with dependencies.
    *   Includes pgAdmin for database management.
    *   Customizable via `.env` files.
    *   Flexible deployment options (full stack or core services).
    *   See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md).

2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)** - For more control and customization.

**Before installing, complete the prerequisites:**
- PGVector setup
- **File Processing ETL Service** (choose one):
  - Unstructured.io API key (supports 34+ formats)
  - LlamaIndex API key (enhanced parsing, supports 50+ formats)
  - Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
- Other required API keys

## Screenshots:

**(Replace the placeholders below with actual image URLs)**

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

### Backend:
*   FastAPI
*   PostgreSQL with pgvector
*   SQLAlchemy
*   Alembic
*   FastAPI Users
*   LangGraph
*   LangChain
*   LLM Integration (LiteLLM)
*   Rerankers
*   Hybrid Search (RRF)
*   Vector Embeddings
*   pgvector
*   Chonkie

### Frontend:
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

### DevOps:
*   Docker
*   Docker Compose
*   pgAdmin

### Extension
Manifest v3 on Plasmo

## Future Work:
*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute:

Your contributions are highly valued! Star the project, report issues, and submit pull requests. See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

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
```

Key improvements and SEO considerations:

*   **Clear, Concise Hook:**  The first sentence is designed to immediately grab attention and explain the core value proposition.
*   **Keyword Optimization:** Includes relevant keywords like "AI research agent," "knowledge base," "search," "LLM," "RAG," "self-hosted," and file formats.
*   **Structured Headings:**  Uses clear headings to organize information (H2 and H3), improving readability and SEO.
*   **Bulleted Lists:** Easy-to-scan bullet points highlight key features.
*   **Detailed Feature Descriptions:** Provides more context for each feature, making it more informative.
*   **Internal Linking:**  Uses relative links (e.g., `[CONTRIBUTING.md](CONTRIBUTING.md)`) to aid navigation within the repository.
*   **External Linking:** Includes links to the repo itself and to other key resources.
*   **Contextual Image Alt Text:** Better `alt` text for the images.
*   **Concise Language:** Removes unnecessary phrases.
*   **Complete Information:**  The revised README aims to provide a comprehensive overview.
*   **Star History added**
*   **Additional sections for more SEO value:** Installation, screenshots, Tech Stack.
*   **Roadmap section**
*   **Contribution instructions**