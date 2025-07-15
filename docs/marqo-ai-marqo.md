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

**Effortlessly build powerful semantic search applications with Marqo, a cutting-edge vector search engine.**

Marqo simplifies vector search by providing an end-to-end solution that handles vector generation, storage, and retrieval through a single API. Built for both text and images, it removes the need to manage complex machine learning models, enabling developers to quickly integrate advanced search capabilities into their projects.  [Explore Marqo on GitHub](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **State-of-the-Art Embeddings:** Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
*   **High Performance:** Achieve rapid search speeds with in-memory HNSW indexes, scalable to millions of documents.
*   **Documents-in, Documents-out:** Simplify your workflow with out-of-the-box vector generation, storage, and retrieval for text and images.
*   **Multimodal Search:** Easily combine text and image search for richer, more relevant results.
*   **Flexible Search Queries:** Build complex semantic queries by combining weighted search terms and using Marqo’s query DSL for filtering.
*   **Managed Cloud Option:** Benefit from low-latency deployments, easy scaling, high availability, and 24/7 support with [Marqo Cloud](https://www.marqo.ai/cloud).

## Getting Started

1.  **Install Docker:** Follow the instructions on the [Docker Official website](https://docs.docker.com/get-docker/) to install Docker and ensure at least 8GB memory and 50GB storage are allocated.

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

**[Further Examples and Operations](#getting-started)**

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **[Haystack](https://github.com/deepset-ai/haystack)**
*   **[Griptape](https://github.com/griptape-ai/griptape)**
*   **[Langchain](https://github.com/langchain-ai/langchain)**
*   **[Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More

*   [Quick start](#Getting-started)
*   [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)
*   [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)
*   [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering)
*   [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs)
*   [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)
*   [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)
*   [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)
*   [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)
*   [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)
*   [Features](#-Core-Features)

## Production Deployment

For production environments, consider deploying Marqo using the provided Kubernetes templates ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)). Alternatively, leverage the fully managed Marqo Cloud service ([https://cloud.marqo.ai](https://cloud.marqo.ai)).

## Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Support

*   [Discourse forum](https://community.marqo.ai)
*   [Slack community](https://bit.ly/marqo-community-slack)

## Contribute

Contribute to the Marqo project by reading the [CONTRIBUTING.md](./CONTRIBUTING.md) document.

## Dev Setup and Merge Instructions

Follow the steps in the original README for development setup and merge instructions.