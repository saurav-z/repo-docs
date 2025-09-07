# SurfSense: Your AI-Powered Research Assistant

**SurfSense empowers you to conduct comprehensive research by integrating your personal knowledge base with external sources like search engines, documents, and more.**  [View the original repo](https://github.com/MODSetter/SurfSense).

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **Private Knowledge Base Integration:** Combine your personal files and data sources for a custom research experience.
*   📁 **Extensive File Format Support:** Upload and manage content from various file types (50+ file extensions supported).
*   🔍 **Powerful Search Capabilities:** Quickly find information within your saved content.
*   💬 **Natural Language Chat:** Interact with your data and receive cited answers.
*   📄 **Cited Answers:** Get trustworthy results from your data.
*   🔔 **Privacy & Local LLM Support:**  Works seamlessly with local LLMs like Ollama.
*   🏠 **Self-Hosted Solution:** Open-source and easy to deploy on your own infrastructure.
*   🎙️ **AI-Powered Podcasts:**
    *   Rapid podcast generation (3-minute podcast in under 20 seconds).
    *   Convert chat conversations into audio.
    *   Supports local (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs
    *   Supports 6000+ Embedding Models.
    *   Supports all major Rerankers (Pinecode, Cohere, Flashrank etc)
    *   Hierarchical Indices (2 tiered RAG setup).
    *   Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **External Data Connectors:** Integrate with a wide array of sources, including:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Airtable, Google Calendar and more.
*   🔗 **Cross-Browser Extension:** Save webpages directly into your SurfSense knowledge base.

## Supported File Extensions

SurfSense supports a wide range of file types.  Note that file support varies based on the ETL service you configure:

*   **Documents & Text:** .pdf, .doc, .docx, .docm, .dot, .dotm, .rtf, .txt, .xml, .epub, .odt, .wpd, .pages, .key, .numbers, .602, .abw, .cgm, .cwk, .hwp, .lwp, .mw, .mcw, .pbd, .sda, .sdd, .sdp, .sdw, .sgl, .sti, .sxi, .sxw, .stw, .sxg, .uof, .uop, .uot, .vor, .wps, .zabw (LlamaCloud), and more
*   **Presentations:** .ppt, .pptx, .pptm, .pot, .potm, .potx, .odp, .key
*   **Spreadsheets & Data:** .xlsx, .xls, .xlsm, .xlsb, .xlw, .csv, .tsv, .ods, .fods, .numbers, .dbf, .123, .dif, .sylk, .slk, .prn, .et, .uos1, .uos2, .wk1, .wk2, .wk3, .wk4, .wks, .wq1, .wq2, .wb1, .wb2, .wb3, .qpw, .xlr, .eth
*   **Images:** .jpg, .jpeg, .png, .gif, .bmp, .svg, .tiff, .webp, .html, .htm, .web
*   **Audio & Video:** .mp3, .mpga, .m4a, .wav, .mp4, .mpeg, .webm
*   **Email & Communication:** .eml, .msg, .p7s

## Installation

SurfSense offers flexible installation options:

1.  **Docker Installation (Recommended):**  The easiest way to get started, with all dependencies containerized. [Docker Installation](https://www.surfsense.net/docs/docker-installation)
2.  **Manual Installation:** For users who prefer greater control. [Manual Installation](https://www.surfsense.net/docs/manual-installation)

Both methods have detailed, OS-specific instructions for Windows, macOS, and Linux.

## Screenshots

*   [Research Agent](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)
*   [Search Spaces](https://github.com/user-attachments/assets/e254c38c-f937-44b6-9e9d-770db583d099)
*   [Manage Documents](https://github.com/user-attachments/assets/7001e306-eb06-4009-89c6-8fadfdc3fc4d)
*   [Podcast Agent](https://github.com/user-attachments/assets/6cb82ffd-9e14-4172-bc79-67faf34c4c1c)
*   [Agent Chat](https://github.com/user-attachments/assets/bb352d52-1c6d-4020-926b-722d0b98b491)
*   [Browser Extension](https://github.com/user-attachments/assets/1f042b7a-6349-422b-94fb-d40d0df16c40)

## Tech Stack

### BackEnd

*   FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LiteLLM integration, Rerankers, Hybrid Search, Vector Embeddings, pgvector, Chonkie

### FrontEnd

*   Next.js, React, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table

### DevOps

*   Docker, Docker Compose, pgAdmin

### Extension

*   Manifest v3 on Plasmo

## Roadmap and Future Development

SurfSense is under active development.  [Check out our public roadmap](https://github.com/users/MODSetter/projects/2) to stay updated on new features and contribute your ideas! Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to get involved.

## Contribute

We welcome contributions of all sizes!  Review the [CONTRIBUTING.md](CONTRIBUTING.md) file for detailed guidelines.

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>