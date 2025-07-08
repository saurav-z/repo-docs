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

**Marqo simplifies vector search by handling vector generation, storage, and retrieval through a single API, enabling developers to build powerful search applications quickly.**  Explore the official [Marqo repository](https://github.com/marqo-ai/marqo) for more details.

**Key Features:**

*   **State-of-the-Art Embeddings:** Utilize cutting-edge machine learning models from PyTorch, Hugging Face, OpenAI, and more, with CPU and GPU support.
*   **Blazing-Fast Performance:** Experience rapid search speeds with in-memory HNSW indexes, scaling to hundreds of millions of documents with horizontal index sharding, and async data handling.
*   **Documents-in, Documents-Out:** Simplify your workflow with built-in vector generation, storage, and retrieval for both text and images, easily building search, entity resolution, and data exploration applications.
*   **Multimodal Search:** Seamlessly combine text and image search with CLIP models, searching using images and text with ease.
*   **Flexible Search Capabilities:** Implement complex semantic queries, incorporate weighted search terms, and filter search results using Marqo’s query DSL.
*   **Comprehensive Integrations:** Integrates with popular AI and data processing frameworks, including Haystack, Griptape, Langchain and Hamilton.
*   **Managed Cloud Option:** Benefit from low-latency optimized deployments, scalable inference, high availability, 24/7 support, and access control through [Marqo Cloud](https://www.marqo.ai/cloud).

## Quick Start Guide

Get up and running with Marqo in minutes:

1.  **Prerequisites:** Marqo requires Docker. Install Docker from the [official website](https://docs.docker.com/get-docker/), ensuring at least 8GB memory and 50GB storage is allocated.
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

4.  **Start Indexing and Searching:**

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2")  # Example: using the e5-base-v2 model

    mq.index("my-first-index").add_documents([
        {
            "Title": "The Travels of Marco Polo",
            "Description": "A 13th-century travelogue describing Polo's travels"
        }, 
        {
            "Title": "Extravehicular Mobility Unit (EMU)",
            "Description": "The EMU is a spacesuit that provides environmental protection, "
                           "mobility, life support, and communications for astronauts",
            "_id": "article_591"
        }],
        tensor_fields=["Description"]
    )

    results = mq.index("my-first-index").search(
        q="What is the best outfit to wear on the moon?"
    )
    ```

    *   `mq` is the Marqo client.
    *   `create_index()` creates an index.  Specify a model like `hf/e5-base-v2` for text search.
    *   `add_documents()` adds documents to the index.  `tensor_fields` specifies fields for vector indexing.
    *   `search()` performs the search.

## Core Features

*   **State-of-the-Art Embeddings:** Access the latest machine learning models from diverse sources. Pre-configured models or bring your own with CPU and GPU support.
*   **Performance:** Rapid search speeds using in-memory HNSW indexes. Scale to large document indexes with horizontal index sharding. Async and non-blocking data upload and search.
*   **Documents-in-documents-out:** Vector generation, storage, and retrieval provided out of the box for text and images. Build applications with ease.
*   **Managed Cloud:** Access optimized deployments with scaled inference, high availability, and 24/7 support.

## Integrations

Marqo integrates with several popular AI and data processing frameworks.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Use Marqo as your Document Store for Haystack pipelines, such as retrieval-augmentation and question answering.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Leverages Marqo's open source or custom models to deliver relevant search results to your LLMs.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Integrates Marqo with LangChain applications, including Retrieval QA and Conversational Retrieval QA.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Integrates Marqo with Hamilton LLM applications.

## Learn More

| Topic | Description |
| ------------- | ------------- |
| 📗 [Quick Start](#quick-start) | Build your first application quickly |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo. |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Enhance GPT with Marqo for a knowledge base. |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Generate and categorize images using semantic search. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Preprocess audio for Q&A with Marqo and ChatGPT. |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Image moderation tools. |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Get started with Marqo Cloud. |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Explore an e-commerce demo. |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chatbot with Marqo and OpenAI's ChatGPT API. |
| 🦾 [Features](#-Core-Features) | Marqo's core features. |

## Getting Started

1.  **Prerequisites:** Docker must be installed.  See the [Docker Official website](https://docs.docker.com/get-docker/)
2.  **Run Marqo using Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your container keeps getting killed, increase Docker's memory allocation to at least 6GB (8GB recommended).*
3.  **Install the Marqo client:**

    ```bash
    pip install marqo
    ```
4.  **Start indexing and searching!**
    *   See the [Quick Start](#quick-start) section.
    *   More examples of other basic operations can be found in the original README.

## Running Marqo in Production

Marqo supports Kubernetes templates for cloud providers. Find the repo at [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).

For a managed cloud service, see [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

Access comprehensive Marqo documentation [here](https://docs.marqo.ai/).

## Warning

Do not run other applications on Marqo's Vespa cluster.

## Contributors

Marqo is a community project. Read the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

## Dev Setup

1.  Create a virtual env:  `python -m venv ./venv`
2.  Activate the virtual environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`

## Merge Instructions

1.  Run the full test suite with `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   [Discourse forum](https://community.marqo.ai): Ask questions and share creations.
*   [Slack community](https://bit.ly/marqo-community-slack): Chat with other community members.