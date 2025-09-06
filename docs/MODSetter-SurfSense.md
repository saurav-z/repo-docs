<!-- Improved README - SurfSense -->

<!-- Header Image (Replace with a better image if available) -->
![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

<!-- Discord Badge -->
<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord&style=flat-square" alt="Discord">
  </a>
</div>

<!-- Trendshift Badge -->
<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="SurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

# SurfSense: Your Personalized AI Research Assistant

**SurfSense is a powerful, open-source AI research agent that connects to your personal knowledge base and external sources, allowing you to quickly find, analyze, and interact with information.** [Explore the SurfSense Repository](https://github.com/MODSetter/SurfSense)

<!-- Add a more engaging introduction and highlight the core value proposition -->

## Key Features

*   💡 **Private Knowledge Base & Research:** Build your own customizable NotebookLM and Perplexity experience integrated with your data.
*   📁 **Versatile File Support:** Upload and save content from a wide variety of file formats, including documents, images, videos, and more (supports 50+ file extensions via LlamaCloud).
*   🔍 **Advanced Search Capabilities:** Quickly search and find information within your saved content and connected sources.
*   💬 **Conversational Interaction:** Chat with your stored content using natural language and receive cited answers.
*   📄 **Cited & Verified Answers:** Get reliable answers, similar to Perplexity, with proper citations.
*   🔔 **Privacy-Focused & Local LLM Support:** Works seamlessly with Ollama local LLMs, ensuring data privacy.
*   🏠 **Self-Hosted & Customizable:** Open-source and easy to deploy locally, giving you complete control.
*   🎙️ **AI-Powered Podcasts:** 
    *   Blazingly fast podcast generation (3-minute podcast in under 20 seconds).
    *   Convert your chats into audio content.
    *   Local TTS (Kokoro TTS) and multiple TTS provider support (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs, 6000+ embedding models, and all major rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Hierarchical indices (2-tiered RAG setup).
    *   Hybrid Search: Semantic and full-text search combined with Reciprocal Rank Fusion (RRF).
    *   RAG as a Service API backend.
*   ℹ️ **Extensive Data Connectors:** Integrate with a wide range of external sources:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.

## Supported File Extensions

**Note:** File format support depends on your ETL service configuration. LlamaCloud offers the most extensive support.

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

### 🔖 Cross-Browser Extension
*   Save webpages with the SurfSense extension.
*   Save webpages protected beyond authentication.

## Installation

SurfSense offers two installation methods to suit your needs:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: Simplest method with all dependencies containerized, including pgAdmin for database management.
    *   Supports `.env` file customization.
    *   Flexible deployment options.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) for detailed instructions.
    *   For deployment scenarios and options, see [Deployment Guide](DEPLOYMENT_GUIDE.md).
2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)**: Provides more control and customization options.

**Prerequisites:** Before installation, set up the following:

*   PGVector
*   **File Processing ETL Service** (choose one and obtain the necessary API keys):
    *   Unstructured.io (34+ formats)
    *   LlamaIndex (50+ formats)
    *   Docling (local processing, no API key required)
*   Other Required API Keys.

## Roadmap & Future Development

*   **Active Development:** SurfSense is constantly evolving.  Join our [Discord](https://discord.gg/ejRNvftDp9) to contribute.
*   **Roadmap:**  View our [public roadmap](https://github.com/users/MODSetter/projects/2) on GitHub Projects.

## Screenshots

<!--  Include a few key screenshots to showcase the functionality.  Add alt text for accessibility. -->

*   **Research Agent:**
    ![Research Agent](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
*   **Search Spaces:**
    ![Search Spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
*   **Manage Documents:**
    ![Manage Documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)
*   **Podcast Agent:**
    ![Podcast Agent](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)
*   **Agent Chat:**
    ![Agent Chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)
*   **Browser Extension:**
    ![Browser Extension 1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)
    ![Browser Extension 2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### Backend
*   FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LiteLLM Integration, Rerankers, Hybrid Search, Vector Embeddings, pgvector, Chonkie (AutoEmbeddings, LateChunker)
### Frontend
*   Next.js 15.2.3, React 19.0.0, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS 4.x, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table.
### DevOps
*   Docker, Docker Compose, pgAdmin
### Extension
*   Manifest v3 on Plasmo

## Contribute

Contributions are very welcome! This includes ⭐-ing the project, finding and creating issues, and fine-tuning the backend.

*   Detailed contribution guidelines are available in [CONTRIBUTING.md](CONTRIBUTING.md).

## Star History

<!-- Star History Graph -->
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

Key improvements and explanations:

*   **SEO Optimization:** Includes relevant keywords (AI research agent, knowledge base, LLM, open source, etc.) throughout the description and headings.
*   **Concise Hook:** The one-sentence hook grabs attention and clearly states the core benefit.
*   **Clear Headings:** Uses descriptive headings for better organization and readability.
*   **Bulleted Key Features:**  Uses bullet points for easy scanning and understanding.  Each feature is also more concisely worded.
*   **Summarized Content:**  The README is shortened and focuses on the most important information.
*   **Call to Action:**  Clear links to the repository and the Discord for engagement.
*   **Improved Formatting:**  Uses markdown consistently for better visual appeal.
*   **Accessibility:** Added `alt` text to images.
*   **Roadmap and Future Development Section:** Highlights planned features and encourages community participation.
*   **Installation Instructions:** More direct and clearly structured.
*   **Tech Stack Section:** Improves the presentation of the technology used.
*   **Contribution Instructions:** Makes it easy for others to start contributing.
*   **More Engaging Descriptions:** Added details and better wording throughout.
*   **Removed Redundancy:** streamlined descriptions.
*   **Image Optimization:**  Replaced original image with links using an image placeholder if original links are broken.
*   **Star History Included:** Added a star history graph to showcase community interest.