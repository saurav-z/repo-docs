# ApeRAG: Your Production-Ready RAG Platform for Intelligent AI Applications

**Unlock the power of Retrieval-Augmented Generation (RAG) with ApeRAG, a comprehensive platform combining graph RAG, vector search, and advanced AI agents to build sophisticated AI applications.** [Explore the ApeRAG Repository](https://github.com/apecloud/ApeRAG)

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the full platform capabilities with our hosted demo

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

## Key Features

*   **Advanced Index Types:** Leverage five index types (Vector, Full-text, Graph, Summary, and Vision) for multi-dimensional document understanding.
*   **Intelligent AI Agents:** Utilize built-in AI agents with MCP support for automated collection selection, intelligent content search, and web search capabilities.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization for a more refined knowledge graph.
*   **Multimodal Processing & Vision Support:** Process and analyze diverse content, including images, charts, and visual elements.
*   **Hybrid Retrieval Engine:** Experience a sophisticated retrieval system that combines Graph RAG, vector search, full-text search, and vision-based search.
*   **MinerU Integration:** Utilize an advanced document parsing service powered by MinerU, offering superior parsing for complex documents with optional GPU acceleration.
*   **Production-Grade Deployment:** Deploy easily with full Kubernetes support, including Helm charts and KubeBlocks integration, for scalable and reliable production environments.
*   **Enterprise Management:** Access built-in audit logging, LLM model management, graph visualization, and comprehensive document and agent workflow management.
*   **MCP Integration:** Integrate seamlessly with AI assistants and tools via Model Context Protocol (MCP) for direct knowledge base access and intelligent querying.
*   **Developer Friendly:** Enjoy a FastAPI backend, React frontend, async task processing with Celery, and a well-documented development environment for easy customization.

## Quick Start

Get started with ApeRAG using Docker Compose:

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

**Access:**

*   Web Interface: [http://localhost:3000/web/](http://localhost:3000/web/)
*   API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

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

**Important:** Replace the URL and API key with your actual ApeRAG API URL and a valid API key.

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG to Kubernetes for high availability and scalability.

**Prerequisites:**

*   Kubernetes cluster (v1.20+)
*   `kubectl`
*   Helm v3+

**Steps:**

1.  **Clone Repository:** `git clone https://github.com/apecloud/ApeRAG.git`
2.  **Deploy Database Services:**  Choose from two options:
    *   **Option A: Use existing databases:**  Configure your database connection details in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy databases with KubeBlocks:** Navigate to `deploy/databases/`, then follow the instructions within the README, including `00-config.sh`, `01-prepare.sh`, and `02-install-database.sh`.
3.  **Deploy ApeRAG Application:**  If using Option B, database connections are pre-configured. If using existing databases, configure them in `deploy/aperag/values.yaml`.

```bash
helm install aperag ./deploy/aperag --namespace default --create-namespace
```

**Access:**

```bash
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default
```

*   Web Interface: [http://localhost:3000](http://localhost:3000)
*   API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

For production environments, configure Ingress in `values.yaml`.

## Development & Contributing

Refer to our [Development Guide](./docs/development-guide.md) for detailed setup instructions and information on contributing to ApeRAG.

## Acknowledgments

ApeRAG is built upon excellent open-source projects:

*   **LightRAG:** Graph-based knowledge retrieval, a modified version of [LightRAG](https://github.com/HKUDS/LightRAG) ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779)).

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-202595.png](docs%2Fimages%2Fstar-history-202595.png)

## License

ApeRAG is licensed under the Apache License 2.0.  See the [LICENSE](./LICENSE) file for details.