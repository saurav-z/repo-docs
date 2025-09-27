# ApeRAG: The Production-Ready RAG Platform for Intelligent AI Applications

**[Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG) | [Try the Live Demo](https://rag.apecloud.com/)**

ApeRAG empowers you to build cutting-edge AI applications with its advanced Retrieval-Augmented Generation (RAG) platform, offering hybrid retrieval, multimodal document processing, intelligent agents, and enterprise-grade management features.

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

## Key Features

*   **Advanced Index Types:** Harness the power of five index types: Vector, Full-text, Graph, Summary, and Vision for comprehensive document understanding and search.
*   **Intelligent AI Agents:** Deploy built-in AI agents with MCP (Model Context Protocol) tool support for automated collection identification, intelligent content search, and web search capabilities.
*   **Enhanced Graph RAG with Entity Normalization:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization (entity merging) for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing & Vision Support:** Unlock multimodal document processing, including vision capabilities for images, charts, and visual content analysis.
*   **Hybrid Retrieval Engine:** Utilize a sophisticated retrieval system combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search.
*   **MinerU Integration:** Leverage an advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.
*   **Production-Grade Deployment:** Enjoy full Kubernetes support with Helm charts and KubeBlocks integration for simplified deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management:** Access built-in audit logging, LLM model management, graph visualization, a comprehensive document management interface, and agent workflow management.
*   **MCP Integration:** Seamlessly integrate with AI assistants and tools through full support for Model Context Protocol (MCP).
*   **Developer-Friendly:** Benefit from a FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and an agent development framework.

## Quick Start

Get started with ApeRAG quickly using Docker Compose:

**Prerequisites:**
* CPU >= 2 Core
* RAM >= 4 GiB
* Docker & Docker Compose

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
docker-compose up -d --pull always
```

**Access:**

*   **Web Interface:** http://localhost:3000/web/
*   **API Documentation:** http://localhost:8000/docs

**MCP (Model Context Protocol) Support:** Configure your MCP client:

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
**Important:** Replace `http://localhost:8000` with your actual ApeRAG API URL and `your-api-key-here` with a valid API key.

**Enhanced Document Parsing:** Enable the advanced parsing service:

```bash
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d
```
Or with GPU acceleration:
```bash
DOCRAY_HOST=http://aperag-docray-gpu:8639 docker compose --profile docray-gpu up -d
```
Or with GNU Make:
```bash
make compose-up WITH_DOCRAY=1
make compose-up WITH_DOCRAY=1 WITH_GPU=1
```

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG on Kubernetes for high availability and scalability.

**Prerequisites:**
*   [Kubernetes cluster](https://kubernetes.io/docs/setup/) (v1.20+)
*   [`kubectl`](https://kubernetes.io/docs/tasks/tools/)
*   [Helm v3+](https://helm.sh/docs/intro/install/)

**Steps:**

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/apecloud/ApeRAG.git
    cd ApeRAG
    ```

2.  **Deploy Database Services:**

    *   **Option A: Use Existing Databases:** Configure in `deploy/aperag/values.yaml`.
    *   **Option B: Deploy with KubeBlocks:**
        ```bash
        cd deploy/databases/
        # Optional: Review configuration - edit 00-config.sh
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

4.  **Access Your Deployment:**
    ```bash
    kubectl port-forward svc/aperag-frontend 3000:3000 -n default
    kubectl port-forward svc/aperag-api 8000:8000 -n default
    # Web Interface: http://localhost:3000
    # API Documentation: http://localhost:8000/docs
    ```

**Configuration Options:**
Review `values.yaml` for advanced settings.  By default includes the `doc-ray` service, to disable set `docray.enabled: false`

**Troubleshooting:**
See `deploy/databases/README.md` for database management. Check pod logs with:

```bash
kubectl logs -f deployment/aperag-api -n default
kubectl logs -f deployment/aperag-frontend -n default
```

## Acknowledgments

ApeRAG leverages and builds upon these open-source projects:

### LightRAG
Modified for production use, the graph-based knowledge retrieval in ApeRAG is powered by [LightRAG](https://github.com/HKUDS/LightRAG)

*   **Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
*   **Authors**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
*   **License**: MIT License

See our [LightRAG modifications changelog](./aperag/graph/changelog.md) for details.

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)

<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-2025922.png](docs%2Fimages%2Fstar-history-2025922.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file.
```
Key improvements and SEO considerations:

*   **Concise Hook:** The one-sentence hook immediately grabs attention.
*   **Clear Headings:** Organized content with clear, descriptive headings.
*   **Bulleted Key Features:** Easily scannable list of core functionalities.
*   **Keywords:** Incorporated relevant keywords like "RAG," "Retrieval-Augmented Generation," "AI agents," "Knowledge Graph," "Kubernetes," "multimodal," etc.
*   **SEO-Friendly Structure:**  Use of H1, H2, and bold text for emphasis, making it easier for search engines to understand the content.
*   **Call to Action:**  Prominent links to the demo and GitHub repository.
*   **Improved Clarity and Brevity:** Streamlined the text for better readability.
*   **Developer Focus:** Explicitly mentioned the development-friendliness.
*   **Complete and Concise:**  Included all the original information.
*   **Kubernetes Focus:**  Strong emphasis on production-ready deployment.
*   **MCP integration:** Highlighted the importance of this feature.