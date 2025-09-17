# ApeRAG: Build Intelligent AI Applications with Advanced RAG

**ApeRAG is a production-ready Retrieval-Augmented Generation (RAG) platform that empowers you to build sophisticated AI applications with hybrid retrieval, intelligent agents, and enterprise-grade features.** ([View on GitHub](https://github.com/apecloud/ApeRAG))

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)**

**Key Features:**

*   **Hybrid Retrieval Engine:** Combines Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive document understanding.
*   **Advanced Index Types:**  Supports Vector, Full-text, Graph, Summary, and Vision indexes for multi-dimensional document understanding.
*   **Intelligent AI Agents:**  Built-in AI agents with MCP (Model Context Protocol) for automated knowledge discovery and question answering.
*   **Enhanced Graph RAG:**  Modified LightRAG implementation with advanced entity normalization for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing & Vision Support:**  Processes images, charts, and visual content alongside traditional text.
*   **MinerU Integration:**  Advanced document parsing service powered by MinerU for superior handling of complex documents, tables, and formulas.
*   **Production-Grade Deployment:**  Full Kubernetes support with Helm charts and KubeBlocks integration for simplified deployments.
*   **Enterprise Management:** Includes audit logging, LLM model management, graph visualization, comprehensive document management, and agent workflow management.
*   **MCP Integration:**  Full support for Model Context Protocol (MCP), enabling seamless integration with AI assistants and tools for direct knowledge base access and intelligent querying.
*   **Developer Friendly:** Built on a FastAPI backend and React frontend, with async task processing via Celery. Extensive documentation, tests, and an agent development framework ease contribution and customization.

## Quick Start

Get started with ApeRAG quickly using Docker Compose:

**Prerequisites:**

*   CPU >= 2 Core
*   RAM >= 4 GiB
*   Docker & Docker Compose

**Steps:**

1.  Clone the repository:

    ```bash
    git clone https://github.com/apecloud/ApeRAG.git
    cd ApeRAG
    ```
2.  Configure the environment:

    ```bash
    cp envs/env.template .env
    ```
3.  Start the services:

    ```bash
    docker-compose up -d --pull always
    ```

**Access:**

*   **Web Interface:**  `http://localhost:3000/web/`
*   **API Documentation:** `http://localhost:8000/docs`

### MCP (Model Context Protocol) Support

Configure your MCP client to interact with ApeRAG:

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

*Replace* `http://localhost:8000` with your actual ApeRAG API URL and replace `your-api-key-here` with a valid API key from your ApeRAG settings.

### Enhanced Document Parsing

Enable advanced document parsing via the MinerU service:

**Enable with Docker Compose:**

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration 
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

**Enable with Makefile:**

```bash
# Enable advanced document parsing service
make compose-up WITH_DOCRAY=1

# Enable advanced parsing with GPU acceleration (recommended)
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

**For detailed configuration and options, refer to the full [README](https://github.com/apecloud/ApeRAG).**

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG to Kubernetes for high availability and scalability.

### Prerequisites

*   Kubernetes cluster (v1.20+)
*   `kubectl`
*   Helm v3+

### Steps

1.  Clone the repository:

    ```bash
    git clone https://github.com/apecloud/ApeRAG.git
    cd ApeRAG
    ```
2.  **Deploy Database Services (Choose one option):**
    *   **Option A: Use Existing Databases:** Configure database connection details in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy Databases with KubeBlocks:** (Recommended for new deployments)
        ```bash
        cd deploy/databases/
        # (Optional) Review configuration - defaults work for most cases
        # edit 00-config.sh
        # Install KubeBlocks and deploy databases
        bash ./01-prepare.sh          # Installs KubeBlocks
        bash ./02-install-database.sh # Deploys PostgreSQL, Redis, Qdrant, Elasticsearch
        # Monitor database deployment
        kubectl get pods -n default
        cd ../../
        ```
        Wait for all database pods to be in `Running` status before proceeding.

3.  **Deploy ApeRAG Application:**
    ```bash
    # If you deployed databases with KubeBlocks in Step 1, database connections are pre-configured
    # If you're using existing databases, edit deploy/aperag/values.yaml with your connection details

    # Deploy ApeRAG
    helm install aperag ./deploy/aperag --namespace default --create-namespace

    # Monitor ApeRAG deployment
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```

4.  **Access Your Deployment:**

    ```bash
    # Forward ports for quick access
    kubectl port-forward svc/aperag-frontend 3000:3000 -n default
    kubectl port-forward svc/aperag-api 8000:8000 -n default

    # Access in browser
    # Web Interface: http://localhost:3000
    # API Documentation: http://localhost:8000/docs
    ```

    For production environments, configure Ingress in `values.yaml` for external access.

**Refer to the full [README](https://github.com/apecloud/ApeRAG) for more details on configuration, troubleshooting, and advanced settings.**

## Development & Contributing

For developers, refer to our [Development Guide](./docs/development-guide.md) for detailed setup instructions and contributing guidelines.

## Acknowledgments

ApeRAG leverages and builds upon the following open-source projects:

*   **LightRAG**: Graph-based knowledge retrieval (Modified version, see [LightRAG](https://github.com/HKUDS/LightRAG) for original paper and details)
    *   **Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
    *   **Authors**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
    *   **License**: MIT License
    *   [LightRAG modifications changelog](./aperag/graph/changelog.md)

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-202595.png](docs%2Fimages%2Fstar-history-202595.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.