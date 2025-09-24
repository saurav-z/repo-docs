# ApeRAG: Your All-in-One RAG Platform for Intelligent AI Applications

**[Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG) | [Try the Live Demo](https://rag.apecloud.com/)**

Unlock the power of Retrieval-Augmented Generation (RAG) with ApeRAG, a production-ready platform designed to build sophisticated AI applications with hybrid retrieval, multimodal document processing, and intelligent agents.

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

ApeRAG empowers you to create intelligent AI solutions that can autonomously search and reason across your knowledge base, offering a superior approach to Knowledge Graph creation, Context Engineering, and AI agent deployment.

**Key Features:**

*   **Advanced Indexing:** Leverage five index types: Vector, Full-text, Graph, Summary, and Vision for comprehensive data understanding and search.
*   **Intelligent AI Agents:** Integrated AI agents with Model Context Protocol (MCP) support for automated collection selection, intelligent content search, and web search capabilities.
*   **Enhanced Graph RAG:**  Deeply modified LightRAG implementation with advanced entity normalization for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing:** Complete support for multimodal document processing, including vision capabilities for image, chart, and visual content analysis.
*   **Hybrid Retrieval Engine:** A sophisticated retrieval system combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive document understanding.
*   **MinerU Integration:** Advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.
*   **Production-Ready Deployment:** Full Kubernetes support with Helm charts and KubeBlocks integration for simplified deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management:**  Built-in audit logging, LLM model management, graph visualization, comprehensive document management interface, and agent workflow management.
*   **MCP Integration:** Full support for Model Context Protocol (MCP), enabling seamless integration with AI assistants and tools for direct knowledge base access and intelligent querying.
*   **Developer-Friendly:**  Built with FastAPI, React, Celery for async task processing, and an agent development framework to help ease customization and contribution.

## Quick Start

Get up and running with ApeRAG using Docker Compose:

**Prerequisites:** Docker & Docker Compose installed

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
docker-compose up -d --pull always
```

*   **Web Interface:** [http://localhost:3000/web/](http://localhost:3000/web/)
*   **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

**MCP (Model Context Protocol) Configuration:**

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
_Replace placeholders with your actual API URL and API key._

**Enhanced Document Parsing:**

Enable advanced document parsing for improved handling of complex documents.

```bash
# Enable advanced document parsing service
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d

# Enable advanced parsing with GPU acceleration 
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```

## Kubernetes Deployment (Production Recommended)

Deploy ApeRAG to Kubernetes for high availability and scalability using Helm:

**Prerequisites:** Kubernetes cluster, kubectl, Helm v3+

1.  **Clone the Repository:** `git clone https://github.com/apecloud/ApeRAG.git && cd ApeRAG`
2.  **Deploy Database Services:**
    *   **Option A: Existing Databases:** Configure database connection details in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy with KubeBlocks:**
        ```bash
        cd deploy/databases/
        # (Optional) Review configuration - defaults work for most cases
        # edit 00-config.sh

        # Install KubeBlocks and deploy databases
        bash ./01-prepare.sh          # Installs KubeBlocks
        bash ./02-install-database.sh # Deploys PostgreSQL, Redis, Qdrant, Elasticsearch
        cd ../../
        ```
3.  **Deploy ApeRAG Application:**

    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    ```

4.  **Access Your Deployment:**

    ```bash
    kubectl port-forward svc/aperag-frontend 3000:3000 -n default
    kubectl port-forward svc/aperag-api 8000:8000 -n default
    ```

    *   **Web Interface:** [http://localhost:3000](http://localhost:3000)
    *   **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

    Configure Ingress in `values.yaml` for production environments.

## Development & Contributing

Explore the [Development Guide](./docs/development-guide.md) for detailed setup, configuration, and contribution instructions.

## Acknowledgments

ApeRAG is built upon and integrates the following open-source project:

### LightRAG

*   **Paper:** "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
*   **Authors:** Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
*   **License:** MIT License

See our [LightRAG modifications changelog](./aperag/graph/changelog.md) for details on modifications.

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)
<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-2025922.png](docs%2Fimages%2Fstar-history-2025922.png)

## License

ApeRAG is licensed under the Apache License 2.0.  See the [LICENSE](./LICENSE) file for details.