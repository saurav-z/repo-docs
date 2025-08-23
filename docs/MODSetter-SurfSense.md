# SurfSense: Your Customizable AI Research Agent for Personalized Knowledge

**SurfSense empowers you to conduct in-depth research by connecting your personal knowledge base to a variety of external sources.** ([Back to Original Repo](https://github.com/MODSetter/SurfSense))

![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **Personalized AI Research:** Create your own private NotebookLM and Perplexity experience, integrated with your data.
*   📁 **Comprehensive File Support:** Upload and save content from your documents, images, videos, and more, with support for **50+ file extensions.**
*   🔍 **Powerful Search:** Quickly find relevant information within your saved content.
*   💬 **Conversational AI:** Interact with your data in natural language and receive cited answers.
*   📄 **Reliable Citations:** Get answers with citations, just like Perplexity.
*   🔔 **Privacy-Focused & Local LLM Support:** Works seamlessly with Ollama and other local LLMs.
*   🏠 **Self-Hosted:** Open-source and easy to deploy locally.
*   🎙️ **AI-Powered Podcast Agent:**
    *   Generates 3-minute podcasts in under 20 seconds.
    *   Converts chat conversations into audio content.
    *   Supports local (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs and 6000+ Embedding Models.
    *   Integrates with major rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes hierarchical indices for a 2-tiered RAG setup.
    *   Employs hybrid search combining semantic and full-text search with Reciprocal Rank Fusion (RRF).
    *   RAG as a Service API backend.
*   ℹ️ **Extensive External Source Integration:**
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, YouTube, GitHub, Discord, and more coming soon!
*   🔖 **Cross-Browser Extension:** Save webpages beyond authentication.

## Supported File Extensions

> **Note:** File format support depends on your ETL service configuration (LlamaCloud, Unstructured, Docling).

*   **Documents & Text:** PDF, DOC, DOCX, DOCM, DOT, DOTM, RTF, TXT, XML, EPUB, ODT, WPD, PAGES, KEY, NUMBERS, and many more (see full list in original README).
*   **Presentations:** PPT, PPTX, PPTM, POT, POTM, POTX, ODP, KEY
*   **Spreadsheets & Data:** XLSX, XLS, XLSM, XLSB, XLW, CSV, TSV, ODS, FODS, NUMBERS, and many more (see full list in original README).
*   **Images:** JPG, JPEG, PNG, GIF, BMP, SVG, TIFF, WEBP.
*   **Audio & Video:** MP3, MPGA, M4A, WAV, MP4, MPEG, WEBM.
*   **Email & Communication:** EML, MSG, P7S.

## Getting Started

SurfSense offers flexible installation options:

1.  **Docker Installation:** The easiest way to get SurfSense up and running, with all dependencies containerized.  See [Docker Setup Guide](DOCKER_SETUP.md).
2.  **Manual Installation:** For users who prefer more control. See [Manual Installation Guide](https://www.surfsense.net/docs/manual-installation).

### Prerequisites

Before installing, set up:

*   PGVector
*   A File Processing ETL Service (Unstructured.io, LlamaIndex, or Docling)
*   Other required API keys.

## Screenshots

[Include all the screenshot images from the original README here with appropriate alt text]

## Tech Stack

### Backend

*   FastAPI
*   PostgreSQL with pgvector
*   SQLAlchemy
*   Alembic
*   FastAPI Users
*   LangGraph
*   LangChain
*   LiteLLM Integration
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

## Roadmap & Future Work

*   Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to help shape its future.
*   View the public roadmap: [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)
*   Add more connectors.
*   Patch minor bugs.
*   Document Chat **[REIMPLEMENT]**
*   Document Podcasts

## Contribute

Your contributions are highly valued!  See our [CONTRIBUTING.md](CONTRIBUTING.md) file for guidelines.

## Star History

[Include Star History chart using the provided code.]