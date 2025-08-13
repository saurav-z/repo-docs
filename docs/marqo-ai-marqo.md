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

Marqo simplifies vector search by providing an end-to-end solution for vector generation, storage, and retrieval through a single, easy-to-use API - [Explore Marqo on GitHub](https://github.com/marqo-ai/marqo).

### Key Features:

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own custom embeddings.
    *   Supports both CPU and GPU acceleration for optimal performance.
*   **⚡ High Performance:**
    *   Leverages in-memory HNSW indexes for blazing-fast search speeds.
    *   Scales to massive document indexes with horizontal index sharding.
    *   Offers async and non-blocking data upload and search operations.
*   **🌌 Documents-in, Documents-out:**
    *   Handles vector generation, storage, and retrieval out-of-the-box.
    *   Build powerful search, entity resolution, and data exploration applications with text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo's flexible query DSL.
    *   Store unstructured data and rich metadata together in documents, with support for various data types.
*   **🍱 Managed Cloud Option:**
    *   Low-latency, optimized deployment of Marqo on the cloud.
    *   Scale inference with a single click.
    *   High availability for reliable performance.
    *   Includes access control for security.

### Getting Started

1.  **Prerequisites:** Docker is required. Ensure Docker has at least 8GB of memory and 50GB of storage allocated.

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

4.  **Start Indexing and Searching (Example):**

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

    *   `mq`: The client for interacting with the Marqo API.
    *   `create_index()`: Creates a new index; specify a model (e.g., "hf/e5-base-v2"). See [Model Reference](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/)
    *   `add_documents()`: Indexes documents, specifying fields for vectorization (`tensor_fields`).
    *   The results will contain matching documents ranked by relevance, including highlights.

### Other Basic Operations

*   **Get Document:** Retrieve a document by its ID.
    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```

*   **Get Index Stats:** Get information about an index.
    ```python
    results = mq.index("my-first-index").get_stats()
    ```

*   **Lexical Search:** Perform a keyword search.
    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```

*   **Multimodal and Cross-Modal Search:** Combine text and image search using CLIP models (requires index configuration).

    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)

    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])

    results = mq.index("my-multimodal-index").search('animal')
    ```

*   **Searching Using an Image:** Search using an image URL.
    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```

*   **Weighted Queries:** Craft complex queries with weighted terms.

    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```

*   **Multimodal Combination Fields:** Combine text and images within a single field.

*   **Delete Documents:** Remove documents by ID.
    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```

*   **Delete Index:** Remove an index.
    ```python
    results = mq.index("my-first-index").delete()
    ```

### Production Deployment

*   **Kubernetes:**  Deploy Marqo using Kubernetes templates: [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** Fully managed cloud service: [Marqo Cloud](https://cloud.marqo.ai)

### Documentation

*   Comprehensive documentation: [https://docs.marqo.ai/](https://docs.marqo.ai/)

### Community

*   [Discourse forum](https://community.marqo.ai)
*   [Slack community](https://bit.ly/marqo-community-slack)

### Contributing

*   Contribute to Marqo:  [CONTRIBUTING.md](./CONTRIBUTING.md)

### Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`

### Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request.

### Support

*   Ask questions and share your ideas on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack) to chat with other community members.