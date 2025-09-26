# ApeRAG: Build Intelligent AI Applications with Production-Ready RAG

**Unlock the power of advanced AI with ApeRAG, a production-ready Retrieval-Augmented Generation (RAG) platform.** ([View on GitHub](https://github.com/apecloud/ApeRAG))

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the full platform capabilities with our hosted demo

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

ApeRAG empowers you to build sophisticated AI applications by combining cutting-edge technologies:

*   **Hybrid Retrieval:** Leverage Graph RAG, vector search, and full-text search.
*   **Multimodal Processing:** Analyze text, images, charts, and visual content.
*   **Intelligent Agents:** Deploy AI agents that can autonomously search and reason.
*   **Enterprise-Grade Management:** Benefit from built-in audit logging, model management, and more.

ApeRAG is ideal for creating Knowledge Graphs, optimizing Context Engineering, and deploying intelligent AI agents that can autonomously search and reason across your knowledge base.

[阅读中文文档](README-zh.md)

**Key Features:**

*   **Advanced Index Types:** Five comprehensive index types: Vector, Full-text, Graph, Summary, and Vision.
*   **Intelligent AI Agents:** Built-in agents with MCP support for automated content discovery, intelligent search, and web search.
*   **Enhanced Graph RAG with Entity Normalization:** Improved knowledge graphs and relational understanding through advanced entity normalization.
*   **Multimodal Processing & Vision Support:** Complete processing for images, charts, and visual content alongside text.
*   **Hybrid Retrieval Engine:** Combines Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search.
*   **MinerU Integration:** Advanced document parsing with optional GPU acceleration.
*   **Production-Grade Deployment:** Kubernetes support with Helm charts and KubeBlocks integration.
*   **Enterprise Management:** Features include audit logging, LLM model management, and comprehensive document management.
*   **MCP Integration:** Full support for Model Context Protocol, enabling seamless integration with AI assistants.
*   **Developer Friendly:** FastAPI backend, React frontend, async task processing with Celery, comprehensive development guides, and an agent development framework.

## Quick Start

Get up and running quickly with ApeRAG using Docker Compose:

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

### MCP (Model Context Protocol) Support

Integrate with AI assistants using MCP. Configure your MCP client with:

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

Replace placeholders with your actual ApeRAG API URL and API key.

### Enhanced Document Parsing

Enable advanced parsing for complex documents with MinerU:

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration (recommended)
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

See the full documentation for more options.

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG to Kubernetes for high availability and scalability.

### Prerequisites

*   Kubernetes cluster (v1.20+)
*   kubectl
*   Helm v3+

### Step-by-step deployment (using KubeBlocks)

1.  Clone the repository: `git clone https://github.com/apecloud/ApeRAG.git`
2.  Deploy Database Services:
    *   Deploy databases with KubeBlocks:
        ```bash
        cd deploy/databases/
        bash ./01-prepare.sh
        bash ./02-install-database.sh
        cd ../../
        ```
3.  Deploy ApeRAG Application:
    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    ```
4.  Access ApeRAG
    *   Use port forwarding: `kubectl port-forward svc/aperag-frontend 3000:3000 -n default` and `kubectl port-forward svc/aperag-api 8000:8000 -n default`
    *   Web Interface: http://localhost:3000
    *   API Documentation: http://localhost:8000/docs

For detailed instructions, troubleshooting, and database management, consult the full documentation.

## Acknowledgments

ApeRAG builds upon the following open-source project:

*   **LightRAG** by HKUDS (MIT License)
    *   Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" (arXiv:2410.05779)

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-2025922.png](docs%2Fimages%2Fstar-history-2025922.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.