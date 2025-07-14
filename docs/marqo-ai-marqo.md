<p align="center">
<img src="https://github.com/marqo-ai/public-assets/blob/main/marqowbackground.png" width="50%" height="40%">
</p>

<p align="center">
<b><a href="https://www.marqo.ai">Website</a> | <a href="https://docs.marqo.ai">Documentation</a> | <a href="https://demo.marqo.ai">Demos</a> | <a href="https://community.marqo.ai">Discourse</a>  | <a href="https://bit.ly/marqo-community-slack">Slack Community</a> | <a href="https://www.marqo.ai/cloud">Marqo Cloud</a>
</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/unit_test_200gb_CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/marqo-ai/marqo/unit_test_200gb_CI.yml?branch=mainline"></a>
<a align="center" href="https://bit.ly/marqo-community-slack"><img src="https://img.shields.io/badge/Slack-blueviolet?logo=slack&amp;logoColor=white"></a>

## Marqo: The Open-Source Vector Search Engine for Text and Images

**Unlock the power of semantic search with Marqo, an end-to-end vector search engine that simplifies embedding generation, storage, and retrieval for both text and images.**  Find the original repository [here](https://github.com/marqo-ai/marqo).

### Key Features:

*   **🤖 State-of-the-Art Embeddings:**
    *   Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or easily integrate your own custom models.
    *   Enjoy CPU and GPU support for optimal performance.

*   **⚡ High-Performance Search:**
    *   Benefit from in-memory HNSW indexes for blazing-fast search speeds.
    *   Scale to handle indexes with hundreds of millions of documents using horizontal index sharding.
    *   Experience async and non-blocking data upload and search capabilities.

*   **🌌 Documents-in, Documents-Out:**
    *   Simplify your workflow with built-in vector generation, storage, and retrieval.
    *   Build powerful search, entity resolution, and data exploration applications using text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Refine results using Marqo’s query DSL and store unstructured data and metadata.

*   **🍱 Managed Cloud Option:**
    *   Access a low-latency, optimized deployment of Marqo.
    *   Scale inference with ease.
    *   Benefit from high availability, 24/7 support, and access control features.  Learn more [here](https://www.marqo.ai/cloud).

### Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

### Quick Start: Get Started with Marqo

1.  **Install Docker:** Follow the instructions on the [Docker Official website](https://docs.docker.com/get-docker/) to install Docker. Ensure Docker has at least 8GB memory and 50GB storage.
2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```
3.  **Install the Marqo client:**
    ```bash
    pip install marqo
    ```
4.  **Start Indexing and Searching:**  See the example code in the original README for a basic index and search operation.

### Learn More

*   📗 [Quick start](#Getting-started)
*   🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)
*   📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)
*   🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering)
*   🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs)
*   🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)
*   🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)
*   ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)
*   👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)
*   🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)
*   🦾 [Features](#-Core-Features)

### Production Deployment

Marqo provides Kubernetes templates and a fully managed cloud service ([https://cloud.marqo.ai](https://cloud.marqo.ai)) for easy deployment.

### Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

### Contributing

Marqo is a community project.  Contribute by reading the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

### Support

*   Join our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack).

### Dev Setup

*   Create a virtual env ```python -m venv ./venv```.
*   Activate the virtual environment ```source ./venv/bin/activate```.
*   Install requirements from the requirements file: ```pip install -r requirements.txt```.
*   Run tests by running the tox file. CD into this dir and then run "tox".
*   If you update dependencies, make sure to delete the .tox dir and rerun.

### Merge Instructions

1. Run the full test suite (by using the command `tox` in this dir).
2. Create a pull request with an attached github issue.