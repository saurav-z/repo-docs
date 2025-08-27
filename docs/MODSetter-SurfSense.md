<div align="center">
  <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
</div>

<!-- Discord Badge -->
<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord" alt="Discord">
  </a>
</div>

# SurfSense: Your Personalized AI Research Assistant

**SurfSense transforms your research by connecting your personal knowledge base to a network of sources, delivering intelligent insights and streamlined workflows.**  Dive deeper into your data with a customizable AI research agent that integrates seamlessly with your existing tools.  **[Explore SurfSense on GitHub](https://github.com/MODSetter/SurfSense)**

<div align="center">
<a href="https://trendshift.io/repositories/13606" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>


## Key Features

*   💡 **Private AI Research Companion:**  Build your own private, customizable research environment, similar to NotebookLM and Perplexity, but connected to your personal data.
*   📁 **Comprehensive File Support:** Upload and utilize content from documents, images, videos, and many other file formats. Supports **50+ file extensions**.
*   🔍 **Advanced Search Capabilities:** Quickly find anything within your saved content.
*   💬 **Conversational Interaction:**  Chat with your saved content using natural language and receive cited answers.
*   📄 **Cited Answers:**  Get answers with source citations for enhanced reliability.
*   🔔 **Privacy & Local LLM Support:** Seamlessly integrates with local LLMs like Ollama.
*   🏠 **Self-Hosted & Open Source:** Deploy SurfSense locally with ease.
*   🎙️ **AI-Powered Podcast Generation:**
    *   Create engaging audio content from your chats.
    *   Blazingly fast podcast generation (3-minute podcast in under 20 seconds).
    *   Supports local and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG (Retrieval-Augmented Generation) Techniques:**
    *   Supports 100+ LLMs and 6000+ Embedding Models.
    *   Compatible with major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes Hierarchical Indices (2-tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **Extensive External Source Integrations:**
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Google Calendar, and more.
*   🔖 **Cross-Browser Extension:**  Save web pages directly to your knowledge base using the SurfSense extension.

## Supported File Extensions

> **Note:** File format support depends on your ETL service configuration. LlamaCloud supports 50+ formats, Unstructured supports 34+ core formats, and Docling (core formats, local processing, privacy-focused, no API key).

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

### Audio & Video *(Always Supported)*
`.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm`

### Email & Communication
*   **Unstructured:** `.eml`, `.msg`, `.p7s`

## Installation

SurfSense offers two installation options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**:  The easiest way to get SurfSense up and running with all dependencies containerized.
    *   Includes pgAdmin for database management.
    *   Supports environment variable customization.
    *   Flexible deployment options.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) for detailed instructions.
    *   For deployment scenarios and options, see [Deployment Guide](DEPLOYMENT_GUIDE.md)

2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)**:  For users who prefer more control.

Before installation, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:
- PGVector setup
- **File Processing ETL Service** (choose one):
  - Unstructured.io API key (supports 34+ formats)
  - LlamaIndex API key (enhanced parsing, supports 50+ formats)
  - Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
- Other required API keys

## Screenshots

[Include screenshots of the Research Agent, Search Spaces, Manage Documents, Podcast Agent, Agent Chat, and Browser Extension.]

## Tech Stack

### **Backend**

*   FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LLM Integration (LiteLLM), Rerankers, Hybrid Search, Vector Embeddings, Chonkie, AutoEmbeddings, LateChunker.

### **Frontend**

*   Next.js 15.2.3, React 19.0.0, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS 4.x, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table.

### **DevOps**

*   Docker, Docker Compose, pgAdmin.

### **Extension**
 Manifest v3 on Plasmo

## Roadmap & Future Development

**SurfSense is actively evolving!**  Help shape its future:

*   **[View the Roadmap](https://github.com/users/MODSetter/projects/2)**
*   **Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) and contribute your ideas!**

## Contribute

Contributions are highly valued!  You can contribute by finding and creating issues, or by ⭐-ing the project.  
See our [CONTRIBUTING.md](CONTRIBUTING.md) for detailed guidelines.

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>
```
Key improvements and optimization strategies:

*   **SEO Optimization:** Includes the primary keyword ("AI research assistant") in the title and prominently throughout the description. The headings are clear and keyword-rich.
*   **Concise Hook:** The first sentence serves as a clear, compelling hook to grab the reader's attention.
*   **Clear Structure:** Uses headings, subheadings, and bullet points for readability.
*   **Feature Highlighting:** The "Key Features" section is comprehensive and clearly outlines the benefits.
*   **Call to Action:** Encourages users to explore the project on GitHub and contribute.
*   **Links:** Links back to the original repo. Uses more relevant links to documentation.
*   **Conciseness:** Removes redundant information and focuses on the core value proposition.
*   **Technical Detail:** Provides a good overview of the technology stack without overwhelming the reader.
*   **Roadmap & Community:** Highlights the project's roadmap and encourages community participation.
*   **Screenshots:**  Mentioned screenshots to visually engage the user (and added placeholders for them).
*   **Star History:**  Included a chart to visually track the project's popularity (optional, but a good practice).