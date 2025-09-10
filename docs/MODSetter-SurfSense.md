# SurfSense: Your AI Research Agent for Personalized Knowledge Discovery

**Unleash the power of AI to research, analyze, and synthesize information from your personal knowledge base and external sources like never before!**  [Explore SurfSense on GitHub](https://github.com/MODSetter/SurfSense).

[![Discord](https://img.shields.io/discord/1359368468260192417?label=Discord)](https://discord.gg/ejRNvftDp9)

---
## Key Features

*   **Personalized AI Research**: Build your own private, customizable research assistant inspired by NotebookLM and Perplexity.

*   **Multi-Source Integration**: Connect to a wide array of external sources including search engines, Slack, Linear, Jira, ClickUp, Confluence, Gmail, Notion, YouTube, GitHub, Discord, Airtable, Google Calendar, and more.

*   **Extensive File Format Support**: Upload and analyze content from documents, images, videos, and presentations with support for 50+ file extensions.

*   **Powerful Search Capabilities**: Quickly and efficiently search within your saved content to find exactly what you need.

*   **AI-Powered Chat**: Interact with your saved content using natural language and receive cited answers for reliable information.

*   **Local LLM Compatibility**: Seamlessly integrates with local LLMs such as Ollama for enhanced privacy and control.

*   **Self-Hosting Made Easy**: Open-source and easy to deploy locally, giving you full ownership of your data and research process.

*   **Advanced RAG Techniques**: Leveraging cutting-edge RAG (Retrieval-Augmented Generation) for superior results.

    *   Supports 100+ LLMs
    *   6000+ Embedding Models
    *   All major Rerankers (Pinecode, Cohere, Flashrank, etc.)
    *   Hierarchical Indices (2-tiered RAG setup)
    *   Hybrid Search (Semantic + Full Text Search with Reciprocal Rank Fusion)
    *   RAG as a Service API Backend
    
*   **Podcast Generation**:
    *   Blazingly fast podcast creation (under 20 seconds).
    *   Convert chat conversations to audio content.
    *   Supports local TTS providers (e.g., Kokoro TTS).
    *   Supports multiple TTS providers (OpenAI, Azure, Google Vertex AI).

*   **Cross-Browser Extension**: Save and analyze any webpage you like, even those behind authentication.

## Supported File Extensions

SurfSense offers wide-ranging file format support via various ETL services:

*   **Documents & Text**: `.pdf`, `.doc`, `.docx`, `.docm`, `.dot`, `.dotm`, `.rtf`, `.txt`, `.xml`, `.epub`, `.odt`, `.wpd`, `.pages`, `.key`, `.numbers`, and many more (LlamaCloud).  Also includes `.doc`, `.docx`, `.odt`, `.rtf`, `.pdf`, `.xml`, `.txt`, `.md`, `.markdown`, `.rst`, `.html`, `.org`, `.epub` (Unstructured) and `.pdf`, `.docx`, `.html`, `.htm`, `.xhtml`, `.adoc`, `.asciidoc` (Docling).

*   **Presentations**: `.ppt`, `.pptx`, `.pptm`, `.pot`, `.potm`, `.potx`, `.odp`, `.key` (LlamaCloud).  Also includes `.ppt`, `.pptx` (Unstructured) and `.pptx` (Docling).

*   **Spreadsheets & Data**: `.xlsx`, `.xls`, `.xlsm`, `.xlsb`, `.xlw`, `.csv`, `.tsv`, `.ods`, `.fods`, `.numbers`, `.dbf`, and others (LlamaCloud).  Also includes `.xls`, `.xlsx`, `.csv`, `.tsv` (Unstructured) and `.xlsx`, `.csv` (Docling).

*   **Images**: `.jpg`, `.jpeg`, `.png`, `.gif`, `.bmp`, `.svg`, `.tiff`, `.webp`, `.html`, `.htm`, `.web` (LlamaCloud). Also includes `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.heic` (Unstructured) and `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`, `.tif`, `.webp` (Docling).

*   **Audio & Video**: `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm` (Always Supported).

*   **Email & Communication**: `.eml`, `.msg`, `.p7s` (Unstructured).

## Get Started

SurfSense offers flexible installation options:

1.  **[Docker Installation](https://www.surfsense.net/docs/docker-installation)**: The easiest way to deploy SurfSense with all dependencies containerized, including pgAdmin for database management and environment variable customization.  See the [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md) for details.
2.  **[Manual Installation (Recommended)](https://www.surfsense.net/docs/manual-installation)**: For users who prefer more control and customization over their setup.

**Prerequisites**: Before installing, complete the [prerequisite setup steps](https://www.surfsense.net/docs/), including:

*   PGVector setup
*   **File Processing ETL Service** (choose one):
    *   Unstructured.io API key (supports 34+ formats)
    *   LlamaIndex API key (enhanced parsing, supports 50+ formats)
    *   Docling (local processing, no API key required, supports PDF, Office docs, images, HTML, CSV)
*   Other required API keys

---

## Tech Stack Highlights

### Backend

*   **FastAPI**: Modern, fast web framework for building APIs with Python
*   **PostgreSQL with pgvector**: Database with vector search capabilities
*   **SQLAlchemy**: SQL toolkit and ORM for database interactions
*   **Alembic**: A database migrations tool
*   **FastAPI Users**: Authentication and user management
*   **LangGraph & LangChain**: Frameworks for AI agent development
*   **LLM Integration**: Through LiteLLM
*   **Rerankers & Hybrid Search**: Advanced result ranking
*   **Vector Embeddings & pgvector**: For semantic search
*   **Chonkie**: Advanced document chunking

### Frontend

*   **Next.js**: React framework with app router, server components, and optimized rendering
*   **React**: Library for building user interfaces
*   **TypeScript**: Static type-checking for JavaScript
*   **Vercel AI SDK Kit UI Stream Protocol**: Scalable chat UI
*   **Tailwind CSS**: Utility-first CSS framework
*   **Shadcn, Lucide React, Framer Motion, Sonner, Geist**: UI component libraries
*   **React Hook Form & Zod**: Form management and validation
*   **@hookform/resolvers & @tanstack/react-table**: UI component libraries

### DevOps

*   **Docker & Docker Compose**: Containerization for consistent deployment
*   **pgAdmin**: Web-based PostgreSQL administration tool

### Extension

*   Manifest v3 on Plasmo

---
## Join the Community and Contribute

SurfSense is actively evolving! Your contributions are highly valued.

*   Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to shape the future.
*   Explore our [CONTRIBUTING.md](CONTRIBUTING.md) for contribution guidelines.
*   Contribute by ⭐ or raising issues.

## Roadmap

Stay updated with our development progress and upcoming features. View the [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2).

---

<p align="center">
    <img 
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4" 
      alt="Catalyst Project" 
      width="200"
    />
</p>
```
Key improvements and summaries:

*   **SEO Optimization**:  Used relevant keywords (AI research agent, personalized knowledge base, LLM, self-hosting, etc.) throughout the text.
*   **Clear Structure**:  Used headings, subheadings, and bullet points for readability and SEO.
*   **Concise Language**: Rephrased and streamlined information for clarity.
*   **One-Sentence Hook**:  The opening sentence is designed to capture attention and define the project's purpose.
*   **Call to Action**:  Encourages users to explore the project and join the community.
*   **Comprehensive Feature List**:  Expanded and organized the key features.
*   **Complete Installation Instructions**:  Clear and detailed installation instructions are included.
*   **Tech Stack Section**: More concise tech stack descriptions.
*   **Contribution Guidelines**: Includes how to contribute.
*   **Links back to GitHub**: Added multiple links for SEO purposes.
*   **Roadmap Link**: Included to show activity.
*   **Removed Unnecessary Details**: Removed details that were not required and kept the information as concise as possible.

This improved README is more informative, user-friendly, and optimized for search engines, making it easier for potential users to discover and understand SurfSense.