<div align="center">
  <img src="https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65" alt="SurfSense Header">
</div>

<div align="center">
  <a href="https://discord.gg/ejRNvftDp9">
    <img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">
  </a>
</div>

# SurfSense: Your Customizable AI Research Agent

**SurfSense empowers you to conduct in-depth research by connecting to your personal knowledge base and a wealth of external sources, making information discovery effortless.** ([View Original Repo](https://github.com/MODSetter/SurfSense))

<div align="center">
  <a href="https://trendshift.io/repositories/13606" target="_blank">
    <img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/>
  </a>
</div>

## Key Features

*   **Private & Customizable Knowledge Base:** Leverage your own data with the power of AI, similar to NotebookLM and Perplexity, but fully tailored to your needs.
*   **Extensive File Format Support:** Upload and manage content from a variety of file formats (50+ supported via LlamaCloud, 34+ via Unstructured, core formats via Docling), including documents, images, and videos.
*   **Powerful Search Capabilities:** Quickly find relevant information within your saved content with advanced search features.
*   **Interactive Chat with Your Data:** Engage in natural language conversations with your saved content and receive cited answers for accurate information.
*   **Cited Answers:** Get verifiable, cited answers, just like Perplexity.
*   **Local LLM Support & Privacy:** Seamlessly integrates with local LLMs like Ollama for enhanced privacy.
*   **Self-Hostable:** Open-source and easy to deploy locally.
*   **Blazing-Fast Podcast Generation:**
    *   Generate 3-minute podcasts in under 20 seconds.
    *   Convert chat conversations into audio content.
    *   Supports local TTS (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Employs Hierarchical Indices (2-tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **Browser Extension:** Save any webpage with the SurfSense extension.

## Supported Integrations (External Sources)

SurfSense connects to a wide range of sources to enhance your research:

*   Search Engines (Tavily, LinkUp)
*   Slack
*   Linear
*   Jira
*   ClickUp
*   Confluence
*   Notion
*   Gmail
*   YouTube Videos
*   GitHub
*   Discord
*   Airtable
*   Google Calendar
*   And more coming soon!

## Supported File Extensions

SurfSense offers broad file format support. The specific formats supported depend on your chosen ETL service configuration:

*   **Documents & Text:** (PDF, DOCX, RTF, TXT, and many more - see the original README for full lists for LlamaCloud, Unstructured and Docling.)
*   **Presentations:** (PPTX, PPT, KEY, and more)
*   **Spreadsheets & Data:** (XLSX, CSV, ODS, and more)
*   **Images:** (JPG, PNG, GIF, BMP, SVG, and more)
*   **Audio & Video:** (MP3, MP4, WEBM, and more)
*   **Email & Communication:** (EML, MSG, and more)

## Get Started with SurfSense

### Installation Options

Choose the method that best suits your needs:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)** - The easiest method, with all dependencies containerized.
    *   Includes pgAdmin for database management via a web UI.
    *   Uses environment variables for customization via a `.env` file.
    *   Flexible deployment options (full stack or core services only).
    *   Simplified configuration.
    *   See [Docker Setup Guide](DOCKER_SETUP.md) for detailed instructions.
    *   For deployment scenarios and options, see [Deployment Guide](DEPLOYMENT_GUIDE.md)

2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)** - For users who prefer more control or need custom deployments.

Before installing, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:

*   PGVector setup
*   **File Processing ETL Service:** Choose one:
    *   Unstructured.io API key (34+ formats)
    *   LlamaIndex API key (enhanced parsing, 50+ formats)
    *   Docling (local processing, no API key, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

## Screenshots

<details>
<summary>Click to expand Screenshots</summary>

**Research Agent**

<img src="https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4" alt="Research Agent">

**Search Spaces**

<img src="https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099" alt="Search Spaces">

**Manage Documents**

<img src="https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d" alt="Manage Documents">

**Podcast Agent**

<img src="https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c" alt="Podcast Agent">

**Agent Chat**

<img src="https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491" alt="Agent Chat">

**Browser Extension**

<img src="https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40" alt="Browser Extension">
<img src="https://github.com/user-attachments/assets/a9b9f1aa-2677-404d-b0a0-c1b2dddf24a7" alt="Browser Extension">

</details>

## Tech Stack

### Backend

*   FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LLM Integration, Rerankers, Hybrid Search, Vector Embeddings, pgvector, Chonkie.

### Frontend

*   Next.js, React, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table.

### DevOps

*   Docker, Docker Compose, pgAdmin.

### Extension

*   Manifest v3 on Plasmo

## Roadmap & Future Work

*   **Future Work:** Add More Connectors, Bug Fixes, Document Podcasts
*   **Roadmap:**  Stay up-to-date on our development progress and upcoming features! View the [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2).

## Contribute

Contributions are highly encouraged! Star the project, create issues, or submit pull requests to help improve SurfSense. See the [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>

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

*   **Clear, Concise Hook:** The one-sentence hook immediately grabs the reader's attention.
*   **Descriptive Headings:** Uses clear, SEO-friendly headings.
*   **Bulleted Key Features:**  Easy-to-scan bullet points for readability.
*   **Keyword Optimization:** Includes relevant keywords like "AI research agent," "knowledge base," "customizable," "local LLM," "podcast generation," and the specific tools/services it integrates with.
*   **Clear Calls to Action:** Includes links to the original repo, installation docs, and contribution guidelines.
*   **Structured Content:** The use of `details` tag for screenshots.
*   **Internal Linking:**  Links to important resources within the document (e.g., Docker setup, Deployment Guide, Contribution guide).
*   **Concise Summaries:** Avoids overly verbose descriptions.
*   **Emphasis on Benefits:** Highlights the *value* proposition of SurfSense (e.g., effortless information discovery, enhanced privacy).
*   **Mobile-Friendly:**  The use of descriptive alt text ensures images are accessible.

This revised README is now significantly more effective at attracting users, explaining the project's value, and guiding users toward getting started.