# ApeRAG: Build Intelligent AI Applications with Advanced RAG Capabilities

**ApeRAG** is your all-in-one platform for building production-ready Retrieval-Augmented Generation (RAG) applications, combining graph RAG, vector search, and full-text search with advanced AI agents. [Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the power of ApeRAG with our interactive demo.

![ApeRAG Architecture](docs/images/HarryPotterKG2.png)
![Chat Interface](docs/images/chat2.png)

## Key Features

*   **Advanced Index Types:** Leverage five index types (Vector, Full-text, Graph, Summary, Vision) for multi-dimensional document understanding and search.
*   **Intelligent AI Agents:** Utilize built-in AI agents with Model Context Protocol (MCP) for automated knowledge base interaction, intelligent searching, and web search capabilities.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation featuring advanced entity normalization for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing & Vision Support:** Process documents with text, images, charts, and visual content analysis.
*   **Hybrid Retrieval Engine:** Employ a sophisticated retrieval system combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive understanding.
*   **MinerU Integration:** Benefit from an advanced document parsing service powered by MinerU, offering superior parsing for complex documents, tables, and formulas with optional GPU acceleration.
*   **Production-Grade Deployment:** Deploy on Kubernetes using Helm charts and KubeBlocks for high availability and scalability.
*   **Enterprise Management:** Utilize built-in audit logging, LLM model management, graph visualization, document management, and agent workflow management.
*   **MCP Integration:** Integrate seamlessly with AI assistants and tools using Model Context Protocol (MCP) for direct knowledge base access and intelligent querying.
*   **Developer Friendly:** Enjoy a FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and an agent development framework for easy customization and contribution.

## Quick Start

Get started with ApeRAG quickly using Docker Compose:

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

*   **Web Interface:** http://localhost:3000/web/
*   **API Documentation:** http://localhost:8000/docs

## MCP (Model Context Protocol) Support

Integrate with AI assistants using MCP:

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

*   Replace `http://localhost:8000` with your API URL and `your-api-key-here` with a valid API key.
*   MCP Server Provides: Collection browsing, hybrid search, intelligent querying.

## Enhanced Document Parsing

Enable advanced document parsing with MinerU:

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration 
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG for enterprise-grade performance:

**Prerequisites:**

*   Kubernetes cluster (v1.20+)
*   kubectl
*   Helm v3+

**Steps:**

1.  **Clone:** `git clone https://github.com/apecloud/ApeRAG.git && cd ApeRAG`
2.  **Databases:**
    *   **Option A:** Configure existing databases in `deploy/aperag/values.yaml`.
    *   **Option B:** Deploy databases with KubeBlocks:
        ```bash
        cd deploy/databases/
        # (Optional) Review configuration - defaults work for most cases
        # edit 00-config.sh
        bash ./01-prepare.sh
        bash ./02-install-database.sh
        kubectl get pods -n default
        cd ../../
        ```
3.  **Deploy ApeRAG:**
    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```
4.  **Access:**
    ```bash
    kubectl port-forward svc/aperag-frontend 3000:3000 -n default
    kubectl port-forward svc/aperag-api 8000:8000 -n default
    ```

    *   Web Interface: http://localhost:3000
    *   API Documentation: http://localhost:8000/docs

    Configure Ingress in `values.yaml` for production access.

## Development & Contributing

Refer to our [Development Guide](./docs/development-guide.md) for setting up your development environment, advanced configurations, and contributing to ApeRAG.

## Acknowledgments

ApeRAG is built upon:

### LightRAG
Graph-based knowledge retrieval by [LightRAG](https://github.com/HKUDS/LightRAG):
- **Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
- **Authors**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
- **License**: MIT License

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-202595.png](docs%2Fimages%2Fstar-history-202595.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.