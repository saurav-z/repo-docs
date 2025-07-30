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

## Marqo: The Open-Source Vector Search Engine for Text & Images

**Marqo simplifies vector search, offering an end-to-end solution for both text and image data with a single API.**  [Explore the Marqo Repository](https://github.com/marqo-ai/marqo)

### Key Features:

*   **Effortless Vector Search:**  Easily index, store, and retrieve vectors with a single API, simplifying the integration of semantic search.
*   **Embeddings Included:** No need to bring your own embeddings, Marqo handles vector generation out-of-the-box.
*   **State-of-the-Art Models:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more with CPU/GPU support.
*   **High-Performance Search:** Benefit from in-memory HNSW indexes for cutting-edge search speeds, scaling to millions of documents.
*   **Multimodal Capabilities:** Seamlessly search and retrieve both text and image data, enabling powerful cross-modal search applications.
*   **Documents-in-Documents-Out:**  Simplify your workflow; Marqo handles preprocessing, embedding, metadata storage, and deployment.
*   **Flexible Queries:** Build complex semantic queries with weighted search terms and filtering via Marqo's query DSL.
*   **Managed Cloud Option:** Leverage a low-latency, optimized deployment of Marqo with scalability, high availability, and 24/7 support.

### Why Choose Marqo?

Marqo goes beyond a traditional vector database by providing a comprehensive solution that includes ML model management, input preprocessing, and the flexibility to modify search behavior. This all-in-one approach empowers developers to quickly build powerful search applications with minimal effort.

### Quick Start

Follow these steps to get started with Marqo using Docker and Python:

1.  **Install Docker:**  Ensure Docker is installed and has at least 8GB of memory and 50GB of storage allocated.
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
4.  **Index and Search:**

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
    See the full [Getting Started](#getting-started) guide for details.

### Core Features

**(See Key Features section above)**

### Integrations

*   **Haystack**: Integrate Marqo as your Document Store with Haystack pipelines.
*   **Griptape**: Leverage Marqo for scalable search within LLM-based agents.
*   **Langchain**: Utilize open-source or custom models with Marqo for LangChain applications.
*   **Hamilton**: Integrate Marqo for Hamilton LLM applications.

### Learn More

*   [Quick Start](#getting-started) - Build your first application in minutes.
*   [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)
*   [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)
*   [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering)
*   [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs)
*   [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)
*   [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)
*   [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)
*   [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)
*   [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)
*   [Core Features](#-Core-Features) - Comprehensive list of features.

### Getting Started

**(See Quick Start above)**

### Other Basic Operations

*   **Get Document:**  Retrieve a document by its ID using `get_document(document_id="your_id")`.
*   **Get Index Stats:** Get index information with `get_stats()`.
*   **Lexical Search:** Perform keyword searches using `search(..., search_method=marqo.SearchMethods.LEXICAL)`.
*   **Multimodal and Cross-Modal Search:** Create indexes to search across text and images with clip models.
*   **Searching with Images:** Search directly with image URLs or local file paths.
*   **Searching with Weights:** Craft sophisticated queries using dictionaries to specify term weights (positive and negative).
*   **Multimodal Combination Fields:** Combine text and images into one field within an index.
*   **Delete Documents:** Delete documents using `delete_documents(ids=["id1", "id2"])`.
*   **Delete Index:** Delete an index with `delete()`.

### Running Marqo in Production

Deploy Marqo using Kubernetes templates for a cloud provider of your choice ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)).  Or, use [Marqo Cloud](https://cloud.marqo.ai) for a fully managed experience.

### Documentation

Full documentation: [https://docs.marqo.ai/](https://docs.marqo.ai/)

### Support

*   [Discourse forum](https://community.marqo.ai) - Ask questions and connect with the community.
*   [Slack community](https://bit.ly/marqo-community-slack) - Chat and share ideas.

### Contributing

See [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to contribute.

### Dev Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests:  `tox`
5.  If you update dependencies, delete the `.tox` directory and rerun `tox`.

### Merge Instructions:

1.  Run the full test suite (`tox`).
2.  Create a pull request with an attached GitHub issue.