# ApeRAG: Build Intelligent AI Applications with a Production-Ready RAG Platform

[![Trust Score](https://archestra.ai/mcp-catalog/api/badge/quality/apecloud/ApeRAG)](https://archestra.ai/mcp-catalog/apecloud__aperag)

**ApeRAG empowers you to build cutting-edge AI applications by seamlessly integrating Graph RAG, vector search, full-text search, and advanced AI agents.** 

[Explore the ApeRAG project on GitHub](https://github.com/apecloud/ApeRAG) | [Try the Live Demo](https://rag.apecloud.com/)

![HarryPotterKG2.png](docs%2Fimages%2FHarryPotterKG2.png)
![chat2.png](docs%2Fimages%2Fchat2.png)

ApeRAG is a comprehensive Retrieval-Augmented Generation (RAG) platform designed for production environments. It provides a robust foundation for building sophisticated AI applications, including knowledge graphs, context engineering, and intelligent AI agents. With hybrid retrieval, multimodal document processing, and enterprise-grade management features, ApeRAG streamlines the development of AI solutions that can autonomously search and reason across your knowledge base.

**Key Features:**

*   **Advanced Index Types:** Leverage five index types (Vector, Full-text, Graph, Summary, Vision) for comprehensive document understanding and search.
*   **Intelligent AI Agents:** Utilize built-in AI agents with MCP support for automated collection identification, intelligent content searching, and web search capabilities.
*   **Enhanced Graph RAG:** Benefit from a deeply modified LightRAG implementation with advanced entity normalization to improve relational understanding and knowledge graph cleanliness.
*   **Multimodal Processing & Vision Support:** Process diverse document types, including images, charts, and visual content, alongside traditional text.
*   **Hybrid Retrieval Engine:** Combine Graph RAG, vector search, full-text search, summary-based retrieval, and vision-based search for complete document understanding.
*   **MinerU Integration:** Integrate advanced document parsing with optional GPU acceleration for superior handling of complex documents.
*   **Production-Grade Deployment:** Deploy on Kubernetes with Helm charts and KubeBlocks for scalable, high-availability deployments.
*   **Enterprise Management:** Access built-in audit logging, LLM model management, graph visualization, document management, and agent workflow management.
*   **MCP Integration:** Seamlessly integrate with AI assistants and tools through full support for the Model Context Protocol (MCP).
*   **Developer Friendly:** Enjoy a FastAPI backend, React frontend, async task processing (Celery), extensive testing, and development guides for easy customization.

## Quick Start

Get started with ApeRAG using Docker Compose:

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

## Kubernetes Deployment (Recommended for Production)

Deploy ApeRAG on Kubernetes for a production-ready, scalable solution.

**Prerequisites:** Kubernetes cluster, `kubectl`, and Helm v3+ installed.

**Steps:**

1.  **Clone Repository:** `git clone https://github.com/apecloud/ApeRAG.git; cd ApeRAG`
2.  **Deploy Databases:**
    *   **Option A (Use Existing):** Configure database connection details in `deploy/aperag/values.yaml`.
    *   **Option B (Deploy with KubeBlocks):** `cd deploy/databases/; bash ./01-prepare.sh; bash ./02-install-database.sh; cd ../../`  (Requires KubeBlocks)
3.  **Deploy ApeRAG Application:**
    ```bash
    helm install aperag ./deploy/aperag --namespace default --create-namespace
    kubectl get pods -n default -l app.kubernetes.io/instance=aperag
    ```
4.  **Access:**  Forward ports for local access or configure Ingress for production.

See the original README for more detailed instructions, troubleshooting, and configuration options.

## Acknowledgments

ApeRAG is built upon and integrates with the open-source project [LightRAG](https://github.com/HKUDS/LightRAG).

## Community

*   [Discord](https://discord.gg/FsKpXukFuB)
*   [Feishu](docs%2Fimages%2Ffeishu-qr-code.png)
<img src="docs/images/feishu-qr-code.png" alt="Feishu" width="150"/>

## Star History
![star-history-2025922.png](docs%2Fimages%2Fstar-history-2025922.png)

## License

ApeRAG is licensed under the Apache License 2.0. See the [LICENSE](./LICENSE) file for details.