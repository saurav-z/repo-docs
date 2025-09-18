# ApeRAG: Your All-in-One RAG Platform for Next-Gen AI Applications

**Tired of building RAG applications from scratch?** ApeRAG is a production-ready Retrieval-Augmented Generation (RAG) platform that combines powerful search, advanced AI agents, and enterprise-grade features to accelerate your AI development. [Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG)

*   **Live Demo:** [Experience ApeRAG's capabilities firsthand!](https://rag.apecloud.com/)

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)

![chat2.png](docs%2Fimages%2Fchat2.png)

**Key Features:**

*   **Advanced Index Types:** Leverage five index types (Vector, Full-text, Graph, Summary, Vision) for comprehensive document understanding and search.
*   **Intelligent AI Agents:** Integrate built-in AI agents with MCP tool support for autonomous content search, knowledge discovery, and web search capabilities.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization (entity merging) for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing & Vision Support:** Process a variety of content with multimodal document processing, including images, charts, and visual content analysis.
*   **Hybrid Retrieval Engine:** Utilize a sophisticated retrieval system combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search.
*   **MinerU Integration:** Use the advanced document parsing service powered by MinerU for superior parsing of complex documents, tables, formulas, and scientific content.
*   **Production-Grade Deployment:** Deploy easily with full Kubernetes support, Helm charts, and KubeBlocks integration for simplified management of databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management:** Get built-in audit logging, LLM model management, graph visualization, a comprehensive document management interface, and agent workflow management.
*   **MCP Integration:** Seamlessly integrate with AI assistants and tools using full support for Model Context Protocol (MCP).
*   **Developer Friendly:** Enjoy a FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and an agent development framework.

## Quick Start

Get up and running with ApeRAG using Docker Compose:

**Prerequisites:** Docker and Docker Compose installed.

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
docker-compose up -d --pull always
```

*   **Web Interface:** Access ApeRAG at http://localhost:3000/web/
*   **API Documentation:** Explore the API at http://localhost:8000/docs

## MCP (Model Context Protocol) Support

ApeRAG supports [MCP (Model Context Protocol)](https://modelcontextprotocol.io/) for seamless integration with AI assistants. Configure your MCP client with:

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

**Important**: Replace `http://localhost:8000` with your actual ApeRAG API URL and `your-api-key-here` with a valid API key.

## Advanced Document Parsing

ApeRAG integrates with MinerU for enhanced document parsing.

**Enable Advanced Document Parsing:**

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration 
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

## Kubernetes Deployment (Production-Ready)

Deploy ApeRAG to Kubernetes for high availability, scalability, and production-grade management.

### Prerequisites

*   Kubernetes cluster (v1.20+)
*   kubectl
*   Helm v3+

### Steps

1.  **Clone the Repository:** `git clone https://github.com/apecloud/ApeRAG.git`
2.  **Deploy Database Services:**
    *   **Option A: Existing Databases:** Configure connections in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy with KubeBlocks:** Follow the instructions in `deploy/databases/README.md`.
3.  **Deploy ApeRAG Application:** `helm install aperag ./deploy/aperag --namespace default --create-namespace`

### Configuration & Access

*   Configure in `values.yaml` (e.g., Ingress for external access).
*   Access your deployment using port forwarding:

```bash
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default
```

*   **Web Interface:** http://localhost:3000
*   **API Documentation:** http://localhost:8000/docs

## Development & Contributing

Explore our [Development Guide](./docs/development-guide.md) for detailed setup and contribution instructions.

## Acknowledgments

ApeRAG leverages and extends the capabilities of excellent open-source projects, including:

*   **LightRAG**: Modified for graph-based knowledge retrieval. ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
    *   Authors: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
    *   License: MIT License

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.