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

Marqo is a powerful, open-source vector search engine that simplifies the process of building intelligent search applications for both text and images. ([See the original repository](https://github.com/marqo-ai/marqo))

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from Hugging Face, OpenAI, and more, with both CPU and GPU support.
*   **⚡ High Performance:** Experience lightning-fast search speeds with in-memory HNSW indexes, supporting scales up to hundreds of millions of documents with horizontal index sharding.
*   **🌌 Document-in, Document-out:**  Marqo handles vector generation, storage, and retrieval seamlessly, allowing you to build search and exploration applications with your text and images.
*   **🍱 Managed Cloud Option:** Benefit from a fully managed cloud service with optimized deployment, scalable inference, high availability, and 24/7 support.  Learn more at [Marqo Cloud](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal and Cross Modal Search:** Easily search images, text, or combinations with CLIP models.
*   **📚 Integrations:** Integrations with Haystack, Griptape, Langchain and Hamilton
*   **✨ Search with Weights:** Advanced queries consisting of multiple components with weightings towards or against them.
*   **🦾 Multimodal Combination Fields:** Build indexes with multimodal combination fields, combining text and images into one field.

## Quick Start

Get started with Marqo in a few simple steps:

1.  **Install Docker:**  Marqo requires Docker. Download it from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage (recommended).
2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```
3.  **Install the Marqo Client:**

    ```bash
    pip install marqo
    ```
4.  **Index and Search!**

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

## Core Features Deep Dive

*   **Embedding Models:** Use cutting-edge machine learning models.  Choose from pre-configured models or bring your own.
*   **Performance Optimization:**  Benefit from in-memory HNSW indexes for exceptional search speeds.  Scale effortlessly with horizontal index sharding and async operations.
*   **Simplified Data Handling:** Marqo provides an intuitive "documents in, documents out" experience, handling vector generation, storage, and retrieval. Build applications using text and images.
*   **Advanced Querying:** Create sophisticated semantic queries by combining weighted search terms and filter results with Marqo’s query DSL.

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks, allowing for smooth implementation.

*   **Haystack:** Use Marqo as your Document Store for Haystack pipelines such as retrieval-augmentation, question answering, document search and more. ([Haystack integration](https://haystack.deepset.ai/integrations/marqo-document-store))
*   **Griptape:** Access scalable search with your data by using the MarqoVectorStoreDriver ([Griptape integration](https://github.com/griptape-ai/griptape))
*   **Langchain:** Leverage open source or custom fine tuned models through Marqo for LangChain applications. ([Langchain integration](https://github.com/langchain-ai/langchain))
*   **Hamilton:** Leverage open source or custom fine tuned models through Marqo for Hamilton LLM applications.

## Learn More

*   **[Quick Start](#quick-start)** Build your first application with Marqo in under 5 minutes.
*   **[Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization):** Advanced image search with Marqo.
*   **[Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code):** Building a multilingual database in Marqo.
*   **[Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering):** Using Marqo as a knowledge base for GPT.
*   **[Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs):** Combining stable diffusion with semantic search.
*   **[Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing):** Add diarisation and transcription to preprocess audio.
*   **[Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo):** Building advanced image search with Marqo to find and remove content.
*   **[Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud):** Get set up and running with Marqo Cloud.
*   **[Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md):** Build an e-commerce web application.
*   **[Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo):** Build a chat bot application using Marqo and OpenAI's ChatGPT API.
*   **[Features](#-Core-Features)** Marqo's core features.

## Getting Started (Detailed)

1.  **Docker Setup:**  Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB storage.
2.  **Run Marqo:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your `marqo` container gets killed, increase Docker memory to at least 6GB (8GB recommended).*
3.  **Install Client:**

    ```bash
    pip install marqo
    ```
4.  **Example Usage:**

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index") # Optionally specify a model:  model="hf/e5-base-v2"

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
    *   `mq`:  The Marqo client.
    *   `create_index()`: Creates an index.  You can specify a model.  Experiment to find the best model for your data.
    *   `add_documents()`: Add documents to the index. `tensor_fields` specifies the fields to be indexed as vectors.
    *   `_id`:  Optional document ID. Marqo generates one if not provided.
    *   Each hit in results contains the `_highlights` field showing what matched the search query.

## Basic Operations

*   **Get Document:** Retrieve a document by its ID:
    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```
*   **Get Index Stats:** Get index information:
    ```python
    results = mq.index("my-first-index").get_stats()
    ```
*   **Lexical Search:** Perform keyword search:
    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```
*   **Multimodal Search:** Enable image and text search with CLIP models.
    ```python
    settings = {
        "treat_urls_and_pointers_as_images": True,
        "model": "ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)

    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus...",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])

    results = mq.index("my-multimodal-index").search('animal')
    ```
*   **Search Using an Image:**
    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```
*   **Weighted Queries:** Construct more complex queries by assigning weights.
    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```
*   **Multimodal Combination Fields:** Combine text and images in one field for document scoring and search.

    ```python
    #  Example shows retrieval of caption and image pairs using multiple types of queries.
    # Create the mappings, here we define our captioned_image mapping
    # which weights the image more heavily than the caption - these pairs
    # will be represented by a single vector in the index
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7
            }
        }
    }
    # We specify which fields to create vectors for.
    # Note that captioned_image is treated as a single field.
    tensor_fields=["captioned_image"]
    ```
*   **Delete Documents:**
    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```
*   **Delete Index:**
    ```python
    results = mq.index("my-first-index").delete()
    ```

## Production Deployment

Marqo offers robust deployment options for production environments:

*   **Kubernetes:**  Use our Kubernetes templates for deployment on your preferred cloud provider. The templates enable cluster deployment with replicas, multiple storage shards, and inference nodes.  Find the repo [here](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** A fully managed cloud service offering optimized deployments, scalability, high availability, and 24/7 support.  Sign up at [cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Notes

*   **Vespa Cluster:** Do not run other applications on Marqo's Vespa cluster, as Marqo automatically adjusts settings.

## Contributing

We welcome contributions!  Please read our [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the environment: `source ./venv/bin/activate`.
3.  Install requirements: `pip install -r requirements.txt`.
4.  Run tests: `tox`.
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   Join our [Discourse forum](https://community.marqo.ai) to ask questions and share your projects.
*   Connect with the community on our [Slack community](https://bit.ly/marqo-community-slack).