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

## Marqo: The Open-Source, End-to-End Vector Search Engine for Text and Images

Marqo simplifies vector search with a single API for vector generation, storage, and retrieval, enabling developers to easily build powerful search applications.

**[Check out the original repo](https://github.com/marqo-ai/marqo)**

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:**
    *   Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Use pre-configured models or easily bring your own custom models.
    *   Supports both CPU and GPU for optimal performance.
*   **⚡ High Performance:**
    *   Utilizes in-memory HNSW indexes for blazing-fast search speeds.
    *   Scales to handle indexes with hundreds of millions of documents via horizontal sharding.
    *   Offers asynchronous and non-blocking data upload and search capabilities.
*   **🌌 Documents-In, Documents-Out:**
    *   Provides out-of-the-box vector generation, storage, and retrieval.
    *   Build search, entity resolution, and data exploration applications with text and images.
    *   Create complex semantic queries with weighted search terms.
    *   Filter search results using Marqo’s query DSL.
    *   Store both unstructured data and semi-structured metadata together in documents, supporting various data types.
*   **🍱 Managed Cloud Option:**
    *   Optimized Marqo deployments with low latency.
    *   Scale inference with a single click.
    *   High availability and reliability.
    *   Access control for enhanced security.
    *   24/7 support.
    *   Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks, with more integrations in development.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Integrate Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Gives LLM-based agents access to scalable search with your own data.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Leverage open source or custom fine tuned models through Marqo for LangChain applications.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Leverage open source or custom fine tuned models through Marqo for Hamilton LLM applications.

## Get Started with Marqo

**Quick Start:**

1.  **Docker Installation**: Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/), ensuring at least 8GB memory and 50GB storage.

2.  **Run Marqo using Docker**:

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

3.  **Install the Marqo Client:**

    ```bash
    pip install marqo
    ```

4.  **Start Indexing and Searching!**

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2")

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

## Learn More

| Resource | Description |
|---|---|
| 📗 [Quick start](#Get-Started) | Build your first application in minutes. |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Dive into advanced image search capabilities. |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Create multilingual databases using Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Use Marqo as a knowledge base for GPT. |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine semantic search with Stable Diffusion. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Preprocess audio for Q&A with Marqo and ChatGPT. |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Build image search with content moderation features. |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Quickstart guide for Marqo Cloud. |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Build an e-commerce app. |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chatbot application. |
| 🦾 [Core Features](#-Core-Features) | Explore Marqo's core features in detail. |

## Advanced Usage

*   **[Getting started](#Getting-started)** - Learn how to get started with a simple example.
*   **Get document** - Retrieve a document by ID.
*   **Get index stats** - Get information about an index.
*   **Lexical search** - Perform a keyword search.
*   **Multi modal and cross modal search** - Search with images using the CLIP model.
*   **Searching using an image** - Search by providing an image link.
*   **Searching using weights in queries** - Refine queries with weights.
*   **Creating and searching indexes with multimodal combination fields** - Create indexes with multimodal combination fields.
*   **Delete documents** - Delete documents.
*   **Delete index** - Delete an index.

## Running Marqo in Production

Marqo provides Kubernetes templates for deployment on your preferred cloud provider, offering scalable clusters with replicas, storage sharding, and inference nodes; you can find the repo [here](https://github.com/marqo-ai/marqo-on-kubernetes).

For a fully managed cloud service, sign up for [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

For comprehensive information, visit the official [Marqo Documentation](https://docs.marqo.ai/).

## Warning

Do not run other applications on Marqo's Vespa cluster as Marqo manages and adapts the settings on the cluster automatically.

## Contributing

Marqo thrives on community contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to contribute.

## Development Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements:  `pip install -r requirements.txt`
4.  Run tests:  `tox`
5.  Update dependencies, delete .tox dir, and rerun tests.

## Merge Instructions

1.  Run the full test suite using the `tox` command.
2.  Create a pull request with an attached GitHub issue.

## Support

*   Discuss and share on our [Discourse forum](https://community.marqo.ai).
*   Join the [Slack community](https://bit.ly/marqo-community-slack).