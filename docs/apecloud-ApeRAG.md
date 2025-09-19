# ApeRAG: The Production-Ready RAG Platform for Intelligent AI Applications

**[Explore ApeRAG on GitHub](https://github.com/apecloud/ApeRAG)** - Build sophisticated AI applications with hybrid retrieval, multimodal document processing, and intelligent agents.

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the full platform capabilities with our hosted demo

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

ApeRAG is a cutting-edge Retrieval-Augmented Generation (RAG) platform designed for production environments. It empowers you to create intelligent AI applications by seamlessly integrating Graph RAG, vector search, and full-text search, enhanced with advanced AI agents. Key features include hybrid retrieval, multimodal document processing, enterprise-grade management, and a developer-friendly architecture.

**Key Features:**

*   **Advanced Index Types:** Leverage five index types for optimal retrieval: Vector, Full-text, Graph, Summary, and Vision for multi-dimensional document understanding.
*   **Intelligent AI Agents:** Utilize built-in AI agents with Model Context Protocol (MCP) tool support that can automatically identify relevant collections, search content intelligently, and provide web search capabilities for comprehensive question answering.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization (entity merging) for cleaner knowledge graphs and improved relational understanding.
*   **Multimodal Processing & Vision Support:** Process a variety of document types, including images, charts, and visual content.
*   **Hybrid Retrieval Engine:** A sophisticated system combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search.
*   **MinerU Integration:** Advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.
*   **Production-Grade Deployment:** Full Kubernetes support with Helm charts and KubeBlocks integration for simplified deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management:** Access built-in audit logging, LLM model management, graph visualization, comprehensive document management interface, and agent workflow management.
*   **MCP Integration:** Full support for Model Context Protocol (MCP), enabling seamless integration with AI assistants and tools for direct knowledge base access and intelligent querying.
*   **Developer-Friendly:** FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and agent development framework for easy contribution and customization.

## Quick Start

Get up and running quickly with Docker Compose:

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

Configure your MCP client with:

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

*Replace `http://localhost:8000` with your API URL and replace `your-api-key-here` with a valid API key from your ApeRAG settings.*

### Enhanced Document Parsing

Enable enhanced document parsing with:

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

For advanced configurations and contributing, refer to the [Development Guide](./docs/development-guide.md).

## Kubernetes Deployment (Production Recommended)

Deploy ApeRAG in a production-ready environment with Kubernetes.

**Prerequisites:**

*   Kubernetes cluster (v1.20+)
*   kubectl configured
*   Helm v3+ installed

**Steps:**

1.  **Clone Repository:**
    ```bash
    git clone https://github.com/apecloud/ApeRAG.git
    cd ApeRAG
    ```
2.  **Deploy Database Services:**
    *   **Option A (Existing Databases):** Configure database connections in `deploy/aperag/values.yaml`.
    *   **Option B (KubeBlocks):** Deploy databases using provided scripts:
        ```bash
        cd deploy/databases/
        # Review and (optionally) edit 00-config.sh
        bash ./01-prepare.sh
        bash ./02-install-database.sh
        # Monitor deployment and return to project root.
        ```
3.  **Deploy ApeRAG Application:**
    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```

**Configuration Options:**

*   Review `values.yaml` for customization (images, resources, Ingress).
*   Disable `doc-ray` by setting `docray.enabled: false`.

**Access:**

```bash
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default
```

*   **Web Interface:** http://localhost:3000
*   **API Documentation:** http://localhost:8000/docs

Configure Ingress for external access in production.

**Troubleshooting:**

*   Database issues: Refer to `deploy/databases/README.md`.
*   Pod status: Check logs with `kubectl logs`.

## Acknowledgments

ApeRAG leverages:

### LightRAG

Graph-based knowledge retrieval is powered by a modified version of [LightRAG](https://github.com/HKUDS/LightRAG):

*   **Paper**: "LightRAG: Simple and Fast Retrieval-Augmented Generation" ([arXiv:2410.05779](https://arxiv.org/abs/2410.05779))
*   **Authors**: Zirui Guo, Lianghao Xia, Yanhua Yu, Tu Ao, Chao Huang
*   **License**: MIT License

See [LightRAG modifications changelog](./aperag/graph/changelog.md) for details.

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)
    <img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History

![star-history-202595.png](docs%2Fimages%2Fstar-history-202595.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.
```
Key improvements and explanations:

*   **SEO Optimization:** Added relevant keywords (RAG, Retrieval-Augmented Generation, AI, Knowledge Graph, Vector Search, Hybrid Retrieval, AI Agents, Kubernetes, Production-Ready) throughout the content, especially in headings and the introductory sentence.
*   **One-Sentence Hook:** The initial sentence clearly and concisely states the core benefit: building sophisticated AI applications. This is critical for grabbing the reader's attention immediately.
*   **Clear Headings and Structure:** Improved the readability with consistent use of headings and subheadings.
*   **Bulleted Key Features:**  Emphasized the key features using a clear, easy-to-scan bulleted list. This helps users quickly grasp the platform's capabilities.
*   **Concise Language:**  Revised text for better clarity and brevity, removing unnecessary words.
*   **Actionable Instructions:** Maintained the Docker Compose and Kubernetes deployment instructions, making it easy for users to get started.  Added important context (e.g., prerequisites).
*   **Links:**  Ensured all relevant links are included (GitHub, live demo, documentation) for ease of access.
*   **Removed Redundancy:** Eliminated repetitive phrases and streamlined the overall flow.
*   **Emphasis on Production Readiness:** Repeatedly emphasized that ApeRAG is production-ready, a key selling point.
*   **Community Links:** Added the Discord link to make it easier for potential users to connect with the ApeRAG community.
*   **Image Alt Tags:** Added alt tags for the images to improve accessibility and SEO (even though the images themselves don't have textual content, an alt tag is still good practice).
*   **Comprehensive:** Covered all the original content, but improved the clarity and organization.
*   **MCP Highlight:** Emphasized the MCP integration as a major feature.
*   **Docker Compose vs. Kubernetes Flow:** Clarified the steps for each deployment type.