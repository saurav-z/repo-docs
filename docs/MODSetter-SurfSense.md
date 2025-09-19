![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

# SurfSense: Your Customizable AI Research Agent

**SurfSense empowers you to conduct in-depth research by connecting your personal knowledge base to a wide array of external sources, including search engines, Slack, and more.**  [Explore the SurfSense Repository](https://github.com/MODSetter/SurfSense)

---

[![Discord](https://img.shields.io/discord/1359368468260192417?logo=discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **Personalized Research Assistant:** Create your own private, customizable NotebookLM and Perplexity experience.
*   📁 **Extensive File Format Support:** Upload and save content from various file types *(Documents, images, videos, and supports **50+ file extensions**)*.
*   🔍 **Powerful Search Capabilities:** Quickly find information within your saved content.
*   💬 **Interactive Chat Interface:** Engage in natural language conversations with your saved data and get cited answers.
*   📄 **Cited Answers:** Receive answers with source citations, similar to Perplexity.
*   🔔 **Privacy-Focused & Local LLM Support:** Works seamlessly with local LLMs like Ollama.
*   🏠 **Self-Hosting:** Open-source and easily deployable locally.
*   🎙️ **AI-Powered Podcasts:**
    *   Generate podcasts from conversations.
    *   Convert your chat conversations into engaging audio content
    *   Supports local TTS providers (Kokoro TTS)
    *   Support for multiple TTS providers (OpenAI, Azure, Google Vertex AI)
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports major Rerankers.
    *   Utilizes Hierarchical Indices (2-tiered RAG).
    *   Employs Hybrid Search (Semantic + Full Text Search with RRF).
    *   RAG as a Service API Backend.
*   ℹ️ **External Data Integrations:** Connect to and search across multiple sources:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.
*   🌐 **Cross-Browser Extension:** Save and analyze any webpage effortlessly.

## Supported File Extensions

SurfSense supports a wide array of file formats.  Note: File format support depends on your ETL service configuration. Choose from LlamaCloud, Unstructured or Docling.

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

SurfSense offers flexible installation options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest and recommended approach using containerization.
    *   Includes pgAdmin for database management.
    *   Supports environment variable customization.
    *   Flexible deployment options.
    *   [Docker Setup Guide](DOCKER_SETUP.md)
    *   [Deployment Guide](DEPLOYMENT_GUIDE.md)
2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)** - For users who prefer more control.

Before you start, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:

*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key
    *   LlamaIndex API key
    *   Docling (local processing, no API key)
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

### **Backend**

*   **FastAPI:** Web framework for building APIs with Python.
*   **PostgreSQL with pgvector:** Database with vector search capabilities.
*   **SQLAlchemy:** SQL toolkit and ORM for database interactions.
*   **Alembic:** Database migrations tool.
*   **FastAPI Users:** Authentication and user management.
*   **LangGraph:** Framework for developing AI-agents.
*   **LangChain:** Framework for developing AI-powered applications.
*   **LLM Integration:** Integration with LLM models through LiteLLM
*   **Rerankers:** Advanced result ranking.
*   **Hybrid Search:** Combines vector similarity and full-text search using Reciprocal Rank Fusion (RRF).
*   **Vector Embeddings:** Document and text embeddings for semantic search.
*   **pgvector:** PostgreSQL extension for vector similarity operations.
*   **Chonkie:** Advanced document chunking and embedding library.
    *   Uses `AutoEmbeddings` for flexible embedding model selection.
    *   `LateChunker` for optimized document chunking based on embedding model's max sequence length

---

### **Frontend**

*   **Next.js 15.2.3:** React framework featuring App Router, server components, automatic code-splitting, and optimized rendering.
*   **React 19.0.0:** JavaScript library for building user interfaces.
*   **TypeScript:** Static type-checking for JavaScript, enhancing code quality.
*   **Vercel AI SDK Kit UI Stream Protocol:** To create scalable chat UI.
*   **Tailwind CSS 4.x:** Utility-first CSS framework for building custom UI designs.
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set implemented as React components.
*   **Framer Motion:** Animation library for React.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family from Vercel.
*   **React Hook Form:** Form state management and validation.
*   **Zod:** TypeScript-first schema validation with static type inference.
*   **@hookform/resolvers:** Resolvers for using validation libraries with React Hook Form.
*   **@tanstack/react-table:** Headless UI for building powerful tables & datagrids.

---

### **DevOps**

*   **Docker:** Container platform for consistent deployment.
*   **Docker Compose:** Tool for running multi-container Docker applications.
*   **pgAdmin:** Web-based PostgreSQL administration tool included in Docker setup.

---

### **Extension**

*   Manifest v3 on Plasmo

## Future Development

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Get Involved

Contributions are highly encouraged! You can contribute by:
*   Providing a ⭐
*   Finding and creating issues
*   Fine-tuning the Backend
For detailed contribution guidelines, please see our [CONTRIBUTING.md](CONTRIBUTING.md) file.

Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to stay updated and help shape the future of SurfSense!

## Roadmap

Stay up to date with our development progress and upcoming features!
Check out our public roadmap and contribute your ideas or feedback:

**View the Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)

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

*   **Concise Hook:** Starts with a compelling one-sentence description.
*   **Keyword Optimization:**  Uses relevant keywords naturally throughout the text (e.g., "AI research agent," "personal knowledge base," "search engines," "self-hosting").
*   **Clear Headings:**  Uses descriptive headings to structure the content and improve readability.
*   **Bulleted Key Features:**  Lists key features in a bulleted format for easy scanning.
*   **Emphasis on Benefits:** Focuses on what the user *gains* (e.g., "Personalized Research Assistant," "Extensive File Format Support").
*   **Strong Call to Action:** Encourages user engagement (e.g., "Explore the SurfSense Repository," "Get Involved").
*   **Internal Linking:** Includes links to important sections (Installation, Roadmap, Contributing).
*   **External Links:**  Includes links to Discord and Trendshift.
*   **Alt Text on Images:**  Adds alt text to images for accessibility and SEO.
*   **Concise and Readable:**  Streamlines language for clarity.
*   **Complete Installation Instructions:** Added links to the installation and pre-requisites.
*   **Removed redundant text:** streamlined the content.
*   **Tech Stack Section:** Added a dedicated section for the technologies.
*   **Star History:** included the star history diagram.
*   **Consistent Formatting:** Improved and standardized markdown formatting.