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

Marqo is revolutionizing search by providing an all-in-one solution for vector generation, storage, and retrieval, making it easy to build powerful search applications.  Find out more on the [original repo](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more. Start with pre-configured models or bring your own. Offers both CPU and GPU support.
*   **⚡ High Performance:** Leverages in-memory HNSW indexes for blazing-fast search speeds, scalable to hundreds of millions of documents with horizontal index sharding. Supports async and non-blocking data upload and search.
*   **🌌 Documents-in-Documents-out:** Handles vector generation, storage, and retrieval seamlessly. Build search, entity resolution, and data exploration applications for text and images. Supports complex semantic queries and filtering.
*   **🍱 Managed Cloud Option:** Offers low-latency deployment, scalable inference, high availability, 24/7 support, and access control. Learn more at [Marqo Cloud](https://www.marqo.ai/cloud).
*   **🖼️ Multi-Modal Search:** Seamlessly integrates text and image search using CLIP models from Hugging Face.
*   **🤝 Integrations:** Works with popular AI and data processing frameworks like Haystack, Griptape, Langchain, and Hamilton.
*   **⭐ Weighted Queries and Multimodal Combination fields**: Leverage advanced querying methods with weighting, and combination fields for multimodal searches

## Getting Started

Get up and running with Marqo quickly using Docker and the Python client.

1.  **Docker Setup:** Ensure Docker is installed, with at least 8GB memory and 50GB storage allocated.
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

4.  **Index and Search (Example):**

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
            "Description": "The EMU is a spacesuit that provides environmental protection, mobility, life support, and communications for astronauts",
            "_id": "article_591"
        }],
        tensor_fields=["Description"]
    )

    results = mq.index("my-first-index").search(
        q="What is the best outfit to wear on the moon?"
    )
    ```

    *   Explore the returned results, including hits, highlights, and scores.
    *   Experiment with different models using the `model` parameter in `create_index()`.

## Core Operations

*   **Get Document:** `mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** `mq.index("my-first-index").get_stats()`
*   **Lexical Search:** `mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal Search:**
    *   Create index with image support:
        ```python
        settings = {
            "treat_urls_and_pointers_as_images":True,
            "model":"ViT-L/14"
        }
        response = mq.create_index("my-multimodal-index", **settings)
        ```
    *   Add images (URLs or local paths) in documents.
    *   Search by text or image URL: `results = mq.index("my-multimodal-index").search('animal')` or `results = mq.index("my-multimodal-index").search('https://example.com/image.jpg')`
*   **Searching using weights in queries**
*   **Creating and searching indexes with multimodal combination fields**
*   **Delete Documents:** `results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])`
*   **Delete Index:** `results = mq.index("my-first-index").delete()`

## Production Deployment

*   **Kubernetes:**  Deploy Marqo using Kubernetes templates: [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** For a fully managed solution: [Marqo Cloud](https://cloud.marqo.ai)

## Documentation

Comprehensive documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/)

## Community and Support

*   **Discourse Forum:** Get support and share your projects: [Discourse Forum](https://community.marqo.ai)
*   **Slack Community:** Chat with the community: [Slack Community](https://bit.ly/marqo-community-slack)

## Contribute

We welcome contributions! See [CONTRIBUTING.md](https://github.com/marqo-ai/marqo/blob/main/CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`
5.  If updating dependencies, delete `.tox` and rerun.

## Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.