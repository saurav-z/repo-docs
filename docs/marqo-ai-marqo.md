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

**Marqo is a powerful, open-source vector search engine that simplifies building search applications for text and images with its all-in-one API.** ([Back to Original Repo](https://github.com/marqo-ai/marqo))

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Leverages the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more, with CPU and GPU support. Easily bring your own custom models or use preconfigured ones.
*   **⚡ High Performance:** Offers cutting-edge search speeds with embeddings stored in-memory HNSW indexes. Scale efficiently with horizontal index sharding and benefit from async, non-blocking data upload and search.
*   **🌌 Documents-In-Documents-Out:** Simplifies vector generation, storage, and retrieval with a single API. Quickly build applications for search, entity resolution, and data exploration using your text and images. Supports complex semantic queries, weighted search terms, and filtering with a query DSL. Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.
*   **🍱 Managed Cloud Option:** Enjoy a low-latency optimized deployment on Marqo Cloud. Scale inference easily, and benefit from high availability, 24/7 support, and access control. Learn more [here](https://www.marqo.ai/cloud).
*   **🖼 Multimodal Search:** Built-in support for both text and image search using models like CLIP.
*   **🔍 Flexible Search Methods:** Supports lexical (keyword) search in addition to semantic (vector) search.
*   **🧩 Integrations:** Seamlessly integrates with popular AI and data processing frameworks like Haystack, Griptape, Langchain, and Hamilton.

## Quick Start

Get started with Marqo in minutes using Docker and Python.

1.  **Prerequisites:**
    *   Docker. Install from the [Docker Official website](https://docs.docker.com/get-docker/) and ensure at least 8GB memory and 50GB storage.

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

4.  **Index and Search Example:**

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

    # Example of printing the results
    import pprint
    pprint.pprint(results)
    ```

## Other Basic Operations

*   **Get Document:** Retrieve by ID: `result = mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** Get index info: `results = mq.index("my-first-index").get_stats()`
*   **Lexical Search:** Keyword search: `result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal Search:** Search images and text using CLIP models.  Create an index with `treat_urls_and_pointers_as_images:True` and then add documents with image URLs. Search with text or image URLs.
*   **Weighted Queries:** Utilize weighted queries for complex search scenarios.  
*   **Multimodal Combination Fields:** Combine text and images within a single field.

## Running Marqo in Production

Marqo supports Kubernetes templates ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)) for deployment on various cloud providers.

For a fully managed cloud service, sign up for Marqo Cloud: [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Find comprehensive documentation here: [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Warning

Do not run other applications on Marqo's Vespa cluster as Marqo automatically changes and adapts the settings on the cluster.

## Contributing

We welcome contributions!  See [this](./CONTRIBUTING.md) to get started.

## Dev Setup

1.  Create a virtual env `python -m venv ./venv`.
2.  Activate `source ./venv/bin/activate`.
3.  Install requirements `pip install -r requirements.txt`.
4.  Run tests `tox`.
5.  Delete and rerun `tox` if you update dependencies.

## Merge Instructions

1.  Run the full test suite (`tox`).
2.  Create a pull request with a linked GitHub issue.

## Support

*   [Discourse forum](https://community.marqo.ai): Ask questions and share your creations.
*   [Slack community](https://bit.ly/marqo-community-slack): Chat with other users.