<!-- Header and Discord Link -->
<div align="center">
  <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
  <br>
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord" alt="Discord">
  </a>
</div>

<!-- Trendshift Badge -->
<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

# SurfSense: Your Customizable AI Research Agent

**SurfSense empowers you to research anything by connecting your personal knowledge base to a wide range of external sources, including search engines, documents, and web services.** ([View on GitHub](https://github.com/MODSetter/SurfSense))

## Key Features

*   **Personalized Research Assistant:** Build a highly customizable AI agent similar to NotebookLM and Perplexity, tailored to your needs.
*   **Comprehensive Knowledge Base:**
    *   **Multiple File Format Support:** Upload and index content from various personal files including documents, images, videos, and more. Supports 50+ file extensions.
    *   **Fast and Efficient Search:** Quickly locate information within your saved content.
    *   **Conversational Chat:** Interact with your saved content using natural language, receiving cited answers.
*   **Advanced Information Retrieval:**
    *   **Cited Answers:** Receive credible, cited answers like Perplexity.
    *   **Privacy & Local LLM Support:** Seamlessly works with local LLMs such as Ollama.
    *   **Self-Hostable:** Deploy SurfSense easily on your own infrastructure.
*   **Podcast Generation:**
    *   Rapid podcast creation from conversations (3-minute podcast in under 20 seconds).
    *   Convert your chats into engaging audio content.
    *   Support for multiple TTS providers (OpenAI, Azure, Google Vertex AI, and local options).
*   **Robust RAG Technology:**
    *   Supports 100+ LLMs
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecode, Cohere, Flashrank etc)
    *   Uses Hierarchical Indices (2 tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **External Data Integration:** Connect to numerous sources:
    *   Search Engines (Tavily, LinkUp)
    *   Slack
    *   Linear
    *   Jira
    *   ClickUp
    *   Confluence
    *   Notion
    *   YouTube Videos
    *   GitHub
    *   Discord
    *   ...and more to come!
*   **Cross-Browser Extension:** Save and index any webpage using the SurfSense browser extension.

## Supported File Extensions

SurfSense offers extensive file format support, with compatibility depending on your ETL service configuration: LlamaCloud (50+), Unstructured (34+), and Docling (core formats).

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

SurfSense offers two installation options:

1.  **Docker Installation**: The simplest method, with all dependencies containerized, including pgAdmin for database management. See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUID.md).
2.  **Manual Installation**: For users desiring more control.  Detailed OS-specific instructions are provided in the documentation.

**Before installing, complete the prerequisite setup**, including PGVector setup, an ETL service (Unstructured.io, LlamaIndex, or Docling), and other required API keys. See the [SurfSense Documentation](https://www.surfsense.net/docs/) for further details.

## Screenshots

*   **Research Agent**
    ![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
*   **Search Spaces**
    ![search_spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
*   **Manage Documents**
    ![documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)
*   **Podcast Agent**
    ![podcasts](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)
*   **Agent Chat**
    ![git_chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)
*   **Browser Extension**
    ![ext1](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)
    ![ext2](https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7)

## Tech Stack

### BackEnd

*   **FastAPI:** Web framework for building APIs (Python).
*   **PostgreSQL with pgvector:** Database with vector search capabilities.
*   **SQLAlchemy:** SQL toolkit and ORM.
*   **Alembic:** Database migrations tool.
*   **FastAPI Users:** Authentication and user management.
*   **LangGraph:** Framework for developing AI-agents.
*   **LangChain:** Framework for developing AI-powered applications.
*   **LLM Integration:** Integration with LLM models through LiteLLM.
*   **Rerankers:** Advanced result ranking.
*   **Hybrid Search:** Vector similarity and full-text search.
*   **Vector Embeddings:** Document and text embeddings.
*   **pgvector:** PostgreSQL extension for vector operations.
*   **Chonkie:** Document chunking and embedding library using `AutoEmbeddings` and `LateChunker`.

### FrontEnd

*   **Next.js 15.2.3:** React framework.
*   **React 19.0.0:** JavaScript library for UI building.
*   **TypeScript:** Static type-checking.
*   **Vercel AI SDK Kit UI Stream Protocol:** For chat UI.
*   **Tailwind CSS 4.x:** CSS framework.
*   **Shadcn:** Headless components library.
*   **Lucide React:** Icon set.
*   **Framer Motion:** Animation library.
*   **Sonner:** Toast notification library.
*   **Geist:** Font family.
*   **React Hook Form:** Form management and validation.
*   **Zod:** TypeScript schema validation.
*   **@hookform/resolvers:** Resolvers for React Hook Form.
*   **@tanstack/react-table:** UI for building tables.

### DevOps

*   **Docker:** Container platform.
*   **Docker Compose:** Multi-container application tool.
*   **pgAdmin:** PostgreSQL administration tool.

### Extension

*   Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Chat **[REIMPLEMENT]**
*   Document Podcasts

## Contribute

Your contributions are highly valued!  See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  Whether it's a ⭐ or helping with finding issues, your help is appreciated!

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>
```
Key improvements and explanations:

*   **SEO Optimization:**  The description includes relevant keywords like "AI research agent," "knowledge base," "search engine," "documents," "self-hosted," and the supported technologies, increasing search engine visibility. Headings, bolded text, and clear organization aid readability and SEO.
*   **Strong Hook:** The one-sentence hook immediately captures the essence of SurfSense and its value.
*   **Concise and Clear:**  The information is presented efficiently, avoiding unnecessary jargon.
*   **Bullet Points:**  Key features are highlighted using bullet points for easy scanning.
*   **Comprehensive:**  Includes file extension support, screenshots, and the tech stack, providing a complete overview.
*   **Call to Action:** Encourages contribution and directs users to the Discord.
*   **Well-Organized:** Uses headings, subheadings, and spacing for readability.
*   **Accurate:**  The information accurately reflects the original README content, just with improved formatting and clarity.  Includes a link back to the original repo.
*   **Clear Installation Instructions:** Installation instructions were cleaned up for clearer instructions.
*   **Removed redundant information** Several sections, like the initial idea and example of key features, were combined to consolidate the README, and improve formatting.