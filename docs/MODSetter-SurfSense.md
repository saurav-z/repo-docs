# SurfSense: Your Private, Customizable AI Research Assistant

**Unleash the power of AI to supercharge your research and knowledge management.** SurfSense is a cutting-edge, open-source AI research agent that allows you to connect your personal knowledge base with external sources, providing a seamless and efficient research experience. [**Check out the original repository here!**](https://github.com/MODSetter/SurfSense)

[<img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">](https://discord.gg/ejRNvftDp9)

[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   **Private AI Research Powerhouse**: Build your own NotebookLM and Perplexity experience, integrated with your data.
*   **Multi-Format File Support**: Upload and save content from documents, images, videos, and more with support for **50+ file extensions**.
*   **Advanced Search Capabilities**: Quickly research and find information within your saved content.
*   **Natural Language Interaction**: Chat with your saved content and receive cited answers.
*   **Cited Answers**: Get reliable, cited answers just like Perplexity.
*   **Local LLM Support**: Seamlessly works with local LLMs, offering privacy and flexibility.
*   **Self-Hostable**: Open-source and easy to deploy on your own infrastructure.
*   **Podcast Agent**:
    *   Blazingly fast podcast generation (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into engaging audio content.
    *   Supports local TTS providers (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques**:
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Uses Hierarchical Indices (2-tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **Extensive External Source Integrations**:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.
*   **Cross-Browser Extension**: Save any webpage with the SurfSense extension.

## Supported File Extensions

SurfSense supports a wide variety of file formats through various ETL services.

### Documents & Text
**LlamaCloud**: `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, `.602`, `.abw`, `.cgm`, `.cwk`, `.hwp`, `.lwp`, `.mw`, `.mcw`, `.pbd`, `.sda`, `.sdd`, `.sdp`, `.sdw`, `.sgl`, `.sti`, `.sxi`, `.sxw`, `.stw`, `.sxg`, `.uof`, `.uop`, `.uot`, `.vor`, `.wps`, `.zabw`

**Unstructured**: `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub`

**Docling**: `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc`

### Presentations
**LlamaCloud**: `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key`

**Unstructured**: `.ppt`, `.pptx`

**Docling**: `.pptx`

### Spreadsheets & Data
**LlamaCloud**: `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, `.123`, `.dif`, `.sylk`, `.slk`, `.prn`, `.et`, `.uos1`, `.uos2`, `.wk1`, `.wk2`, `.wk3`, `.wk4`, `.wks`, `.wq1`, `.wq2`, `.wb1`, `.wb2`, `.wb3`, `.qpw`, `.xlr`, `.eth`

**Unstructured**: `.xls`, `.xlsx`, `.csv`, `.tsv`

**Docling**: `.xlsx`, `.csv`

### Images
**LlamaCloud**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web`

**Unstructured**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic`

**Docling**: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp`

### Audio & Video *(Always Supported)*
`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication
**Unstructured**: `.eml`, `.msg`, `.p7s`

## Installation

Choose the installation method that best suits your needs:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: The easiest way to get SurfSense up and running.
2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)**: For more control and customization.

**Important:** Before installation, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including PGVector, an ETL service, and other required API keys.

## Screenshots

### Research Agent
![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
### Search Spaces
![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
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

### **Frontend**

*   Next.js
*   React
*   TypeScript
*   Vercel AI SDK
*   Tailwind CSS
*   Shadcn
*   Lucide React
*   Framer Motion
*   Sonner
*   Geist
*   React Hook Form
*   Zod
*   @hookform/resolvers
*   @tanstack/react-table

### **DevOps**

*   Docker
*   Docker Compose
*   pgAdmin

### **Extension**

*   Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute

Contributions are welcome!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

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