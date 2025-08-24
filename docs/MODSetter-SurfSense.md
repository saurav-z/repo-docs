![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)

# SurfSense: Your Customizable AI Research Agent

SurfSense empowers you to conduct in-depth research by integrating with your personal knowledge base and various external sources, acting as a highly customizable AI research assistant.  [View the original repository](https://github.com/MODSetter/SurfSense).

[![Discord Server](https://img.shields.io/discord/1359368468260192417?label=Discord&logo=discord)](https://discord.gg/ejRNvftDp9)
[![Trendshift](https://trendshift.io/api/badge/repositories/13606)](https://trendshift.io/repositories/13606)

## Key Features

*   💡 **Personalized AI Research:** Create your own private, customizable research environment combining the power of tools like NotebookLM and Perplexity.
*   📁 **Broad File Format Support:** Upload and save content from documents, images, videos, and more, with support for **50+ file extensions**.
*   🔍 **Advanced Search Capabilities:** Quickly find information within your saved content with powerful search features.
*   💬 **Interactive Chat Interface:** Engage in natural language conversations with your saved data and receive cited answers.
*   📄 **Reliable Cited Answers:** Get credible, source-backed answers, similar to Perplexity.
*   🔔 **Privacy & Local LLM Support:** Works seamlessly with local LLMs, like Ollama, for enhanced privacy.
*   🏠 **Self-Hosted Solution:** SurfSense is open-source and designed for easy local deployment.
*   🎙️ **AI-Powered Podcasts:**
    *   Rapid podcast generation: Create 3-minute podcasts in under 20 seconds.
    *   Transform chat conversations into engaging audio content.
    *   Supports local TTS providers (Kokoro TTS) and multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   📊 **Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   Compatible with 6000+ Embedding Models.
    *   Integrates with all major rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Utilizes Hierarchical Indices (2-tiered RAG setup).
    *   Employs Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   ℹ️ **Extensive External Source Integration:** Connects to a variety of sources, including:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, and Google Calendar.

## Supported File Extensions

**Note:** File format support depends on your chosen ETL service configuration.  LlamaCloud supports 50+ formats, Unstructured supports 34+ core formats, and Docling (local processing, privacy-focused) supports a selection of formats.

*   **Documents & Text:** (LlamaCloud, Unstructured, Docling)
*   **Presentations:** (LlamaCloud, Unstructured, Docling)
*   **Spreadsheets & Data:** (LlamaCloud, Unstructured, Docling)
*   **Images:** (LlamaCloud, Unstructured, Docling)
*   **Audio & Video:** (.mp3, .mpga, .m4a, .wav, .mp4, .mpeg, .webm)
*   **Email & Communication:** (Unstructured)

## Cross-Browser Extension

*   Save web pages directly into your SurfSense knowledge base using the browser extension.
*   Useful for saving content behind authentication.

## Future Development and Community

SurfSense is actively evolving!  Help shape its future by:

*   Joining the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to share your ideas and feedback.
*   Checking out the [Roadmap](https://github.com/users/MODSetter/projects/2) to stay updated on development and upcoming features.

## Getting Started

### Installation Options

SurfSense offers two installation methods:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: The easiest way to get up and running.
    *   Includes pgAdmin for database management via a web UI.
    *   Uses environment variables for customization.
    *   Flexible deployment options (full stack or core services).
    *   See the [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md).
2.  **[Manual Installation](https://www.surfsense.net/docs/manual-installation)**: For users wanting more control.

**Prerequisites:**
Before installation, make sure to complete the [prerequisite setup steps](https://www.surfsense.net/docs/) including:
- PGVector setup
- **File Processing ETL Service** (choose one):
  - Unstructured.io API key (supports 34+ formats)
  - LlamaIndex API key (enhanced parsing, supports 50+ formats)
  - Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
- Other required API keys

## Screenshots

**(Images would go here - replace the placeholder text below with the actual image links from the original README)**

*   **Research Agent:** (Image)
*   **Search Spaces:** (Image)
*   **Manage Documents:** (Image)
*   **Podcast Agent:** (Image)
*   **Agent Chat:** (Image)
*   **Browser Extension:** (Images)

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
*   Chonkie (`AutoEmbeddings`, `LateChunker`)

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

## Contribute

We welcome contributions!  See our [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.  Consider starting by starring the repo!

## Star History

**(Star History graph would go here - it is generating dynamically, so it's omitted from the static markdown)**