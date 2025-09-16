# ApeRAG: Your All-in-One Platform for Advanced Retrieval-Augmented Generation (RAG)

**[Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG) | 🚀 [Try the Live Demo](https://rag.apecloud.com/)**

ApeRAG empowers you to build sophisticated AI applications with hybrid retrieval, multimodal document processing, intelligent agents, and enterprise-grade management features, all in one platform.

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

## Key Features

*   **Advanced Index Types:** Utilize five comprehensive index types: Vector, Full-text, Graph, Summary, and Vision for multi-dimensional document understanding and search capabilities.
*   **Intelligent AI Agents:** Leverage built-in AI agents with Model Context Protocol (MCP) tool support to automate knowledge discovery, intelligent search, and web search integration.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization (entity merging) for improved relational understanding and cleaner knowledge graphs.
*   **Multimodal Processing & Vision Support:** Process diverse content with complete multimodal document processing, including vision capabilities for images, charts, and visual content analysis.
*   **Hybrid Retrieval Engine:** Experience a sophisticated retrieval system that combines Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive document understanding.
*   **MinerU Integration:** Utilize an advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content, with optional GPU acceleration.
*   **Production-Grade Deployment:** Deploy with ease using full Kubernetes support, Helm charts, and KubeBlocks integration for streamlined deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management:** Access built-in audit logging, LLM model management, graph visualization, a comprehensive document management interface, and agent workflow management features.
*   **MCP Integration:** Seamlessly integrate with AI assistants and tools through full support for Model Context Protocol (MCP) for direct knowledge base access and intelligent querying.
*   **Developer-Friendly:** Leverage a FastAPI backend, React frontend, async task processing with Celery, extensive testing, and a comprehensive development guide for easy contribution and customization.

## Quick Start

Get up and running quickly with Docker Compose.

**Prerequisites:**

*   CPU >= 2 Core
*   RAM >= 4 GiB
*   Docker & Docker Compose

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
docker-compose up -d --pull always
```

*   **Web Interface:** http://localhost:3000/web/
*   **API Documentation:** http://localhost:8000/docs

**MCP (Model Context Protocol) Support:**

Configure your MCP client:

```json
{
  "mcpServers": {
    "aperag-mcp": {
      "url": "https://rag.apecloud.com/mcp",
      "headers": {
        "Authorization": "Bearer your-api-key-here"
      }
    }
  }
}
```

*   **Important:** Replace `http://localhost:8000` with your actual ApeRAG API URL and `your-api-key-here` with a valid API key from your ApeRAG settings.

**Enhanced Document Parsing:**

Enable advanced document parsing service:

```bash
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d
```

Enable advanced parsing with GPU acceleration:

```bash
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

Or use the Makefile shortcuts:

```bash
make compose-up WITH_DOCRAY=1
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG to Kubernetes using Helm charts for high availability and scalability.

**Prerequisites:**

*   [Kubernetes cluster](https://kubernetes.io/docs/setup/) (v1.20+)
*   [`kubectl`](https://kubernetes.io/docs/tasks/tools/) configured
*   [Helm v3+](https://helm.sh/docs/intro/install/) installed

**Steps:**

1.  **Clone Repository:** `git clone https://github.com/apecloud/ApeRAG.git`
2.  **Deploy Database Services:**
    *   **Option A: Use Existing Databases:** Configure in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy with KubeBlocks:**

    ```bash
    cd deploy/databases/
    bash ./01-prepare.sh
    bash ./02-install-database.sh
    kubectl get pods -n default
    cd ../../
    ```

3.  **Deploy ApeRAG Application:**

    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```

**Access:**

```bash
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default
```

*   **Web Interface:** http://localhost:3000
*   **API Documentation:** http://localhost:8000/docs

## Development & Contributing

Refer to the [Development Guide](./docs/development-guide.md) for development setup and contribution instructions.

## Acknowledgments

ApeRAG builds upon:

*   **LightRAG:** For graph-based knowledge retrieval.
    *   Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
    *   Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
    *   License: MIT License
    *   See our [LightRAG modifications changelog](./aperag/graph/changelog.md)

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-202595.png](docs%2Fimages%2Fstar-history-202595.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
```
Key improvements and SEO considerations:

*   **Clear, concise title:** Starts with the core function "ApeRAG" followed by keywords to highlight its purpose.
*   **One-sentence hook:**  Immediately grabs attention and explains the value proposition.
*   **Keyword-rich headings:** Uses relevant terms like "RAG," "Retrieval-Augmented Generation," "Hybrid Retrieval," "Knowledge Graph," etc. to improve search rankings.
*   **Bulleted key features:**  Provides a scannable overview of the platform's capabilities, making it easy for users to quickly understand what ApeRAG offers.
*   **Concise quick start:** Streamlines the initial setup instructions with direct commands.
*   **Kubernetes Deployment section:** Emphasizes the benefits of production-grade deployments.
*   **Clear call to actions:** Encourages users to try the demo and explore the GitHub repository.
*   **Detailed but summarized information:** The important steps and configurations were included, with unnecessary text being removed.
*   **Proper formatting:** Uses Markdown for readability and clarity, including bolding and italics.
*   **Community and License sections:** These are important for establishing credibility and transparency.
*   **Star History section:** Added for visual appeal, with the correct image reference.