[![SurfSense Header](https://github.com/user-attachments/assets/e236b764-0ddc-42ff-a1f1-8fbb3d2e0e65)](https://github.com/MODSetter/SurfSense)

## SurfSense: Your Customizable AI Research Assistant 

**SurfSense** empowers you to conduct in-depth research by connecting your personal knowledge base with external sources, offering a flexible and powerful AI research agent. [Explore SurfSense on GitHub](https://github.com/MODSetter/SurfSense)

<div align="center">
<a href="https://discord.gg/ejRNvftDp9">
<img src="https://img.shields.io/discord/1359368468260192417" alt="Discord">
</a>
</div>

<div align="center">
<a href="https://trendshift.io/repositories/13606" target="_blank"><img src="https://trendshift.io/api/badge/repositories/13606" alt="MODSetter%2FSurfSense | Trendshift" style="width: 250px; height: 55px;" width="250" height="55"/></a>
</div>

### Key Features

*   **💡 Intelligent Knowledge Base:** Create a private, customizable research hub akin to NotebookLM and Perplexity, integrating with your data.
*   **📁 Extensive File Support:**  Upload and analyze content from a vast array of file formats, including documents, images, videos, and more (supports **50+ file extensions**).
*   **🔍 Advanced Search:** Quickly find specific information within your saved content and connected sources.
*   **💬 Conversational Research:** Engage with your knowledge base through natural language chat, receiving cited answers for context.
*   **📄 Cited Answers:** Get reliable answers with citations, similar to Perplexity.
*   **🔔 Privacy-Focused & Local LLM Support:** Works seamlessly with local LLMs such as Ollama, prioritizing your data privacy.
*   **🏠 Self-Hostable:** Open-source and easily deployable locally for complete control.
*   **🎙️ AI-Powered Podcast Generation:** 
    *   Blazingly fast podcast generation (3-minute podcast in under 20 seconds).
    *   Converts chat conversations into engaging audio content.
    *   Supports local TTS providers (Kokoro TTS).
    *   Integrates with multiple TTS providers (OpenAI, Azure, Google Vertex AI).
*   **📊 Advanced RAG Techniques:**
    *   Supports 100+ LLMs.
    *   6000+ Embedding Models.
    *   Supports major Rerankers (Pinecone, Cohere, Flashrank, etc.).
    *   Uses Hierarchical Indices (2-tiered RAG setup).
    *   Utilizes Hybrid Search (Semantic + Full Text Search combined with Reciprocal Rank Fusion).
    *   RAG as a Service API Backend.
*   **ℹ️ External Source Integration:** Connects to a wide range of external sources:
    *   Search Engines (Tavily, LinkUp)
    *   Slack, Linear, Jira, ClickUp, Confluence, Notion, Gmail, YouTube, GitHub, Discord, Google Calendar
    *   And many more integrations coming soon!
*   **🔗 Browser Extension:** Save any webpage you like, even behind authentication.

---

## Video

[Watch a Demo](https://github.com/user-attachments/assets/d9221908-e0de-4b2f-ac3a-691cf4b202da)

## Podcast Sample

[Listen to a Sample Podcast](https://github.com/user-attachments/assets/a0a16566-6967-4374-ac51-9b3e07fbecd7)

---

### Supported File Extensions

**Note:** File format support depends on your ETL service configuration.

*   **Documents & Text:** LlamaCloud, Unstructured, Docling formats supported.
*   **Presentations:** LlamaCloud, Unstructured, Docling formats supported.
*   **Spreadsheets & Data:** LlamaCloud, Unstructured, Docling formats supported.
*   **Images:** LlamaCloud, Unstructured, Docling formats supported.
*   **Audio & Video:** `.mp3`, `.mpga`, `.m4a`, `.wav`, `.mp4`, `.mpeg`, `.webm` are always supported.
*   **Email & Communication:** Unstructured formats supported.

---
---
<p align="center">
  <a href="https://handbook.opencoreventures.com/catalyst-sponsorship-program/" target="_blank" rel="noopener noreferrer">
    <img 
      src="https://github.com/user-attachments/assets/329c9bc2-6005-4aed-a629-700b5ae296b4" 
      alt="Catalyst Sponsorship Program" 
      width="600"
    />
  </a>
</p>
---
---

## Get Involved: Feature Requests and Future Development

SurfSense is under active development.  Join the [SurfSense Discord](https://discord.gg/ejRNvftDp9) to contribute ideas and shape the future.

## Roadmap

Stay up-to-date with progress and upcoming features.  Contribute your ideas and feedback:

**View the Roadmap:** [SurfSense Roadmap on GitHub Projects](https://github.com/users/MODSetter/projects/2)

## Getting Started

### Installation Options

*   **Docker Installation:** The easiest method, with all dependencies containerized. See [Docker Setup Guide](DOCKER_SETUP.md) and [Deployment Guide](DEPLOYMENT_GUIDE.md).
*   **Manual Installation:** For users who prefer greater control.

**Prerequisites:**

*   PGVector setup
*   File Processing ETL Service: Unstructured.io API key, LlamaIndex API key, or Docling.
*   Other required API keys.

### Screenshots

**Research Agent**

![updated_researcher](https://github.com/user-attachments/assets/e22c5d86-f511-4c72-8c50-feba0c1561b4)

**Search Spaces**

![search_spaces](https://github.com/user-attachments/assets/e254c386-f937-44b6-9e9d-770db583d099)

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

### **BackEnd**

*   FastAPI, PostgreSQL with pgvector, SQLAlchemy, Alembic, FastAPI Users, LangGraph, LangChain, LLM Integration (LiteLLM), Rerankers, Hybrid Search, Vector Embeddings, pgvector, Chonkie.

### **FrontEnd**

*   Next.js 15.2.3, React 19.0.0, TypeScript, Vercel AI SDK Kit UI Stream Protocol, Tailwind CSS 4.x, Shadcn, Lucide React, Framer Motion, Sonner, Geist, React Hook Form, Zod, @hookform/resolvers, @tanstack/react-table.

### **DevOps**

*   Docker, Docker Compose, pgAdmin.

### **Extension**

*   Manifest v3 on Plasmo

## Future Work

*   Add More Connectors.
*   Patch minor bugs.
*   Document Podcasts

## Contribute

Contributions are welcome! Help out by starring the repo or creating/finding issues. Fine-tuning the backend is highly encouraged. See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## Star History

<a href="https://www.star-history.com/#MODSetter/SurfSense&Date">
 <picture>
   <source media="(prefers-color-scheme: dark)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date&theme=dark" />
   <source media="(prefers-color-scheme: light)" srcset="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
   <img alt="Star History Chart" src="https://api.star-history.com/svg?repos=MODSetter/SurfSense&type=Date" />
 </picture>
</a>
```
Key improvements and SEO optimizations:

*   **Clear, Concise Hook:**  A strong opening sentence that immediately explains the core value proposition.
*   **Keyword Optimization:**  Used relevant keywords throughout the text, such as "AI research agent," "knowledge base," "customizable," and the names of integrations.
*   **Structured Headings:**  Organized the content with clear headings and subheadings for readability and SEO.
*   **Bulleted Lists:**  Used bulleted lists to highlight key features, making the information easier to scan.
*   **Concise Descriptions:**  Kept descriptions brief and to the point.
*   **Internal Links:**  Included a link back to the original repository.
*   **Calls to Action:** Included a call to action to join the Discord, and contribute, etc.
*   **Roadmap Highlight:** Emphasized the roadmap and contribution opportunities.
*   **Image Alt Tags:** Added alt text to the images to improve accessibility and SEO.
*   **Tech Stack Section:** Made the tech stack easily readable.
*   **Star History**: Added the star history component to make the project more attractive.