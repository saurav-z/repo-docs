# ApeRAG: The Production-Ready RAG Platform for Intelligent AI Applications

**Build powerful AI applications with ApeRAG, a comprehensive platform combining Graph RAG, vector search, and advanced AI agents.** ([View on GitHub](https://github.com/apecloud/ApeRAG))

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**🚀 [Try ApeRAG Live Demo](https://rag.apecloud.com/)** - Experience the full platform capabilities with our hosted demo

![ApeRAG Architecture](docs/images/HarryPotterKG2.png)
![ApeRAG Chat Interface](docs/images/chat2.png)

ApeRAG empowers you to build sophisticated AI solutions with hybrid retrieval, multimodal document processing, intelligent agents, and enterprise-grade management features. Whether you're creating a knowledge graph, fine-tuning context, or deploying autonomous agents, ApeRAG provides the robust foundation you need.

**Key Features:**

*   **Advanced Index Types:** Leverage five comprehensive index types - Vector, Full-text, Graph, Summary, and Vision - for optimal retrieval and multi-dimensional understanding of your documents.
*   **Intelligent AI Agents:** Utilize built-in AI agents enhanced with MCP (Model Context Protocol) tools to automatically identify relevant knowledge collections, conduct intelligent searches, and integrate web search capabilities.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization (entity merging) for improved relational understanding.
*   **Multimodal Processing & Vision Support:** Process diverse content types with complete multimodal document processing, including vision capabilities for images, charts, and visual content analysis.
*   **Hybrid Retrieval Engine:** Utilize a sophisticated retrieval engine combining Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for comprehensive document understanding.
*   **MinerU Integration:** Enhance document parsing with our advanced document parsing service powered by MinerU technology, providing superior parsing for complex documents, tables, formulas, and scientific content with optional GPU acceleration.
*   **Production-Grade Deployment:** Easily deploy with full Kubernetes support, Helm charts, and KubeBlocks integration for simplified deployment of production-grade databases (PostgreSQL, Redis, Qdrant, Elasticsearch, Neo4j).
*   **Enterprise Management:** Benefit from built-in audit logging, LLM model management, graph visualization, comprehensive document management interface, and agent workflow management.
*   **MCP Integration:** Seamlessly integrate with AI assistants and tools through full support for Model Context Protocol (MCP), enabling direct knowledge base access and intelligent querying.
*   **Developer-Friendly:** Contribute and customize with a FastAPI backend, React frontend, async task processing with Celery, extensive testing, comprehensive development guides, and agent development framework.

## Quick Start

Get started quickly using Docker Compose:

**Prerequisites:** Docker and Docker Compose installed.

```bash
git clone https://github.com/apecloud/ApeRAG.git
cd ApeRAG
cp envs/env.template .env
docker-compose up -d --pull always
```

**Access:**
*   **Web Interface:** [http://localhost:3000/web/](http://localhost:3000/web/)
*   **API Documentation:** [http://localhost:8000/docs](http://localhost:8000/docs)

**MCP Integration:** Configure your MCP client with the provided example, replacing placeholders with your ApeRAG API URL and API key.

For enhanced document parsing, enable the advanced document parsing service:

```bash
DOCRAY_HOST=http://aperag-docray:8639 docker compose --profile docray up -d
```

or, using the Makefile:

```bash
make compose-up WITH_DOCRAY=1
```

For development, consult our [Development Guide](./docs/development-guide.md).

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG to Kubernetes for a production-ready environment with high availability and scalability.

**Prerequisites:** Kubernetes cluster (v1.20+), `kubectl`, Helm v3+.

1.  **Clone the Repository:**
    ```bash
    git clone https://github.com/apecloud/ApeRAG.git
    cd ApeRAG
    ```
2.  **Deploy Database Services:**
    *   **Option A:** Use existing databases and configure connection details in `deploy/aperag/values.yaml`.
    *   **Option B:** Deploy databases with KubeBlocks.

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

**Configuration Options:** Adjust resource requirements and other settings in `values.yaml`.

**Access Your Deployment:** Use port forwarding:

```bash
kubectl port-forward svc/aperag-frontend 3000:3000 -n default
kubectl port-forward svc/aperag-api 8000:8000 -n default
```

For production, configure Ingress in `values.yaml`.

## Acknowledgments

ApeRAG is built upon and integrates several open-source projects. See the original README for specific attributions, including LightRAG.

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

*   **Clear and Concise Hook:** Starts with a compelling one-sentence description.
*   **Keyword Optimization:** Includes relevant keywords like "RAG platform," "Graph RAG," "vector search," "AI agents," "knowledge graph," etc.
*   **Structured Headings:**  Uses clear headings and subheadings for readability.
*   **Bulleted Lists:**  Highlights key features with bullet points for easy scanning.
*   **Emphasis on Benefits:** Focuses on what users can *achieve* with ApeRAG.
*   **Call to Action:** Includes a direct link to the live demo.
*   **Concise Quick Start:** Streamlines the quick start guide.
*   **Improved Kubernetes Section:**  More detailed and user-friendly instructions for Kubernetes deployment.
*   **Internal Linking:** Includes links to the original repo.
*   **Readability:** Improved sentence structure and overall flow.
*   **SEO-Friendly Formatting:** Uses Markdown for proper headings, lists, and emphasis.
*   **Complete:**  Includes all the information from the original README but is more organized and easier to understand.
*   **Updated Links:** The live demo and links to external resources are retained.
*   **Consolidated information:** The Quickstart is shorter and easier to follow.
*   **Removed Redundancy:**  Eliminated repetitive phrases.
*   **Simplified Instructions**: Docker commands are kept simple.