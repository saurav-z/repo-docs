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

**Marqo empowers developers to build powerful and scalable vector search applications for both text and image data with an easy-to-use API.**  Visit the [Marqo Repository](https://github.com/marqo-ai/marqo) for the source code.

**Key Features:**

*   **Seamless Vector Generation:**  Built-in support for state-of-the-art embedding models, including those from Hugging Face, OpenAI, and more, eliminates the need to bring your own embeddings.
*   **High Performance:**  Leverages in-memory HNSW indexes for blazing-fast search speeds, and scales to millions of documents.
*   **Documents-In, Documents-Out:**  Simplify your workflow with out-of-the-box vector generation, storage, and retrieval.  Index and search your text and images directly.
*   **Multimodal Search:**  Enable cross-modal search by indexing and searching with images and text.
*   **Flexible Search Capabilities:**  Build complex semantic queries by combining weighted search terms, filter results, and store various data types with metadata.
*   **Managed Cloud Option:**  Take advantage of a fully managed cloud service for low-latency deployments, scalable inference, high availability, and 24/7 support.

## Core Features in Detail

### 🤖 State of the Art Embeddings

*   Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more.
*   Start with a pre-configured model or bring your own.
*   CPU and GPU support.

### ⚡ Performance

*   Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
*   Scale to hundred-million document indexes with horizontal index sharding.
*   Async and non-blocking data upload and search.

### 🌌 Documents-in-documents-out

*   Vector generation, storage, and retrieval are provided out of the box.
*   Build search, entity resolution, and data exploration application with using your text and images.
*   Build complex semantic queries by combining weighted search terms.
*   Filter search results using Marqo’s query DSL.
*   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

### 🍱 Managed cloud

*   Low latency optimised deployment of Marqo.
*   Scale inference at the click of a button.
*   High availability.
*   24/7 support.
*   Access control.
*   Learn more [here](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack):**  Use Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape):**  Access scalable search for LLM-based agents.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain):** Integrate Marqo for vector search components within LangChain applications.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Integrate Marqo with Hamilton LLM applications.

## Quick Start Guide

Follow these steps to get started with Marqo:

1.  **Install Docker:**  Ensure you have Docker installed.  See the [Docker Official website](https://docs.docker.com/get-docker/) for instructions.  Allocate at least 8GB of memory and 50GB of storage to Docker.

2.  **Run Marqo using Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

3.  **Install the Marqo client:**

    ```bash
    pip install marqo
    ```

4.  **Start indexing and searching!**

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

    *   `mq` is the client object.
    *   `create_index()` creates an index. You can specify a model (e.g., `"hf/e5-base-v2"`).
    *   `add_documents()` adds documents to the index.  `tensor_fields` specifies fields to be indexed as vectors.
    *   Use the `_id` field to set a document ID.

    Let's view the results:

    ```python
    import pprint
    pprint.pprint(results)
    ```

    *   Each hit corresponds to a matched document.
    *   `_highlights` show the matching parts.
    *   The `_score` indicates relevance.
    *   `limit` is the maximum number of results.

## Other Basic Operations

### Get Document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```
### Get Index Stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi-modal and Cross Modal Search

1.  Create an index with a CLIP configuration:

    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```

2.  Add images within documents:

    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus...",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```

3.  Search the image field using text:

    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```

### Searching Using an Image

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching Using Weights in Queries

```python
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

mq.create_index("my-weighted-query-index")

mq.index("my-weighted-query-index").add_documents(...)

query = {
  "I need to buy a communications device, what should I get?": 1.1,
  "The device should work like an intelligent computer.": 1.0,
}

results = mq.index("my-weighted-query-index").search(q=query)
```

### Creating and Searching Indexes with Multimodal Combination Fields

```python
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}

mq.create_index("my-first-multimodal-index", **settings)

mq.index("my-first-multimodal-index").add_documents(...)

# We specify which fields to create vectors for.
# Note that captioned_image is treated as a single field.
tensor_fields=["captioned_image"]

# Search this index with a simple text query
results = mq.index("my-first-multimodal-index").search(
  q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes."
)
```

### Delete Documents

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo in Production

*   **Kubernetes:**  Marqo provides Kubernetes templates. Find them at [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:**  For a fully managed experience, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Notes

*   **Vespa Cluster:** Do not run other applications on Marqo's Vespa cluster, as Marqo manages its settings.

## Contributing

Marqo is a community project. Contributions are welcome!  Please see [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests:  `tox`

    *   Delete `.tox` and rerun if dependencies are updated.

## Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   **Discourse:** Ask questions and share creations on our [Discourse forum](https://community.marqo.ai).
*   **Slack:**  Join our [Slack community](https://bit.ly/marqo-community-slack) for discussions.