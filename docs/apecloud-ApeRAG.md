# ApeRAG: The Production-Ready RAG Platform for Intelligent AI Applications

**[Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG)** - Unlock the power of advanced AI with ApeRAG, a comprehensive platform for building, deploying, and managing Retrieval-Augmented Generation (RAG) applications.

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the full platform capabilities with our hosted demo

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

ApeRAG empowers you to build sophisticated AI applications by combining Graph RAG, vector search, and full-text search with intelligent AI agents. Ideal for creating custom knowledge graphs, advanced context engineering, and deploying AI assistants, ApeRAG offers a robust and scalable solution for your AI needs.

**Key Features:**

*   **Advanced Index Types**: Leverage five powerful index types: Vector, Full-text, Graph, Summary, and Vision for multi-dimensional document understanding and search.
*   **Intelligent AI Agents**: Utilize built-in AI agents that leverage the Model Context Protocol (MCP) to automatically identify relevant collections, conduct intelligent searches, and even access web resources for comprehensive question answering.
*   **Enhanced Graph RAG with Entity Normalization**: Benefit from a deeply modified LightRAG implementation featuring advanced entity normalization (entity merging) for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing & Vision Support**: Process documents with full multimodal capabilities, including vision processing for images, charts, and visual content analysis alongside traditional text processing.
*   **Hybrid Retrieval Engine**: Harness a sophisticated retrieval system that combines Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive document understanding.
*   **MinerU Integration**: Utilize an advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.
*   **Production-Grade Deployment**: Deploy with confidence using full Kubernetes support, including Helm charts and KubeBlocks integration for simplified deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management**: Access built-in audit logging, LLM model management, graph visualization, a comprehensive document management interface, and agent workflow management for streamlined operations.
*   **MCP Integration**: Seamlessly integrate with AI assistants and tools through full support for the Model Context Protocol (MCP), enabling direct knowledge base access and intelligent querying.
*   **Developer Friendly**: Leverage a FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and an agent development framework for easy customization and contribution.

## Quick Start

Get up and running with ApeRAG quickly using Docker Compose:

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

Access ApeRAG:

*   **Web Interface**: http://localhost:3000/web/
*   **API Documentation**: http://localhost:8000/docs

## MCP (Model Context Protocol) Integration

Integrate AI assistants with your knowledge base using MCP. Configure your MCP client with:

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

**Important**: Replace `http://localhost:8000` with your actual ApeRAG API URL and `your-api-key-here` with a valid API key from your ApeRAG settings.

## Enhanced Document Parsing

Enable advanced document parsing with MinerU for superior performance:

<details>
<summary><strong>Enhanced Document Parsing Commands</strong></summary>

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration 
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

Or use the Makefile shortcuts:
```bash
# Enable advanced document parsing service
make compose-up WITH_DOCRAY=1

# Enable advanced parsing with GPU acceleration (recommended)
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

</details>

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG in a production-ready environment using Kubernetes and Helm.

**Prerequisites:**

*   Kubernetes cluster (v1.20+)
*   kubectl
*   Helm v3+

**Steps:**

1.  Clone the repository:

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
```

2.  Deploy Database Services: Choose from the options below:

    *   **Option A: Use existing databases**: Configure database connections in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy databases with KubeBlocks**:

```bash
cd deploy/databases/
bash ./01-prepare.sh
bash ./02-install-database.sh
kubectl get pods -n default
cd ../../
```

3.  Deploy ApeRAG Application:

```bash
helm install aperag ./deploy/aperag --namespace default --create-namespace
kubectl get pods -n default -l app.kubernetes.io/instance=aperag
```

4.  Access Your Deployment:

```bash
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default
```

Access the web interface at `http://localhost:3000` and the API documentation at `http://localhost:8000/docs`.  For production, configure Ingress in `values.yaml`.

## Development & Contributing

Explore the [Development Guide](./docs/development-guide.md) for detailed setup instructions and contribution guidelines.

## Acknowledgments

ApeRAG builds upon the work of several excellent open-source projects, including:

*   **LightRAG**: (Modified implementation)
    *   Paper: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
    *   Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
    *   License: MIT License
    *   [LightRAG modifications changelog](./aperag/graph/changelog.md)

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)
<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-202595.png](docs%2Fimages%2Fstar-history-202595.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.