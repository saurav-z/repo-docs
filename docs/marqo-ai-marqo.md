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

**Marqo simplifies vector search by providing an end-to-end solution for vector generation, storage, and retrieval, allowing you to easily build powerful search applications.** This README provides a comprehensive overview of Marqo's features, installation, and usage. Learn more about Marqo on the [original GitHub repository](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.  Bring your own or start with pre-configured models. Supports CPU and GPU.
*   **⚡ Blazing Performance:** Leverage in-memory HNSW indexes for fast search speeds. Scale to hundreds of millions of documents with index sharding and enjoy async/non-blocking data handling.
*   **🌌 Documents-In, Documents-Out:** Generate, store, and retrieve vectors seamlessly. Build search, entity resolution, and data exploration applications using text and images. Combine weighted search terms and use Marqo’s query DSL. Store unstructured data and semi-structured metadata together in documents.
*   **🍱 Managed Cloud Option:** Benefit from low-latency deployment, scalable inference, high availability, and 24/7 support. Access control is included.
*   **🖼️ Multimodal Search:** Easily perform search across both text and images using CLIP models.
*   **🦾 Advanced Querying:** Implement weighted queries for nuanced results and negations.
*   **📦 Integrations:**  Seamlessly integrate with Haystack, Griptape, Langchain, and Hamilton.

### Quick Start

Get up and running with Marqo in a few simple steps:

1.  **Install Docker:** Marqo requires Docker. Follow the [Docker Official website](https://docs.docker.com/get-docker/) instructions to install it. Ensure Docker has at least 8GB memory and 50GB storage.
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
4.  **Start Indexing and Searching:**
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
    Refer to [Getting Started](#getting-started) for detailed explanations of the code snippet.

### Core Features

**(See the "Key Features" bullet points for an introduction)**

### Integrations
**(See the "Key Features" bullet points for an introduction)**

### Learn More about Marqo

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

### Getting Started

**(See Quick Start for basic setup)**

*   `mq` is the client that wraps the `marqo` API.
*   `create_index()` creates a new index with default settings. You have the option to specify what model to use. For example, `mq.create_index("my-first-index", model="hf/e5-base-v2")` will create an index with the default text model `hf/e5-base-v2`. Experimentation with different models is often required to achieve the best retrieval for your specific use case. Different models also offer a tradeoff between inference speed and relevancy. See [here](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) for the full list of models.
*   `add_documents()` takes a list of documents, represented as python dicts for indexing. `tensor_fields` refers to the fields that will be indexed as vector collections and made searchable.
*   You can optionally set a document's ID with the special `_id` field. Otherwise, Marqo will generate one.

Let's have a look at the results:

```python
# let's print out the results:
import pprint
pprint.pprint(results)

{
    'hits': [
        {
            'Title': 'Extravehicular Mobility Unit (EMU)',
            'Description': 'The EMU is a spacesuit that provides environmental protection, mobility, life support, and'
                           'communications for astronauts',
            '_highlights': [{
                'Description': 'The EMU is a spacesuit that provides environmental protection, '
                               'mobility, life support, and communications for astronauts'
            }],
            '_id': 'article_591',
            '_score': 0.61938936
        },
        {
            'Title': 'The Travels of Marco Polo',
            'Description': "A 13th-century travelogue describing Polo's travels",
            '_highlights': [{'Title': 'The Travels of Marco Polo'}],
            '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
            '_score': 0.60237324
        }
    ],
    'limit': 10,
    'processingTimeMs': 49,
    'query': 'What is the best outfit to wear on the moon?'
}
```

*   Each hit corresponds to a document that matched the search query.
*   They are ordered from most to least matching.
*   `limit` is the maximum number of hits to be returned. This can be set as a parameter during search.
*   Each hit has a `_highlights` field. This was the part of the document that matched the query the best.

### Other Basic Operations

*   **Get document:** Retrieve a document by ID.
*   **Get index stats:** Get information about an index.
*   **Lexical search:** Perform a keyword search.
*   **Multimodal and cross modal search:** Power image and text search using CLIP models from HuggingFace. You can use urls from the internet (for example S3) or from the disk of the machine.
*   **Searching using an image:** Searching using an image can be achieved by providing the image link.
*   **Searching using weights in queries:** Queries can also be provided as dictionaries where each key is a query and their corresponding values are weights.
*   **Creating and searching indexes with multimodal combination fields:** Marqo lets you have indexes with multimodal combination fields.

### Delete documents

Delete documents.

```python

results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])

```

### Delete index

Delete an index.

```python
results = mq.index("my-first-index").delete()
```

### Running Marqo in Production

For production deployments, use the Kubernetes templates available in the [marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes) repository. For a fully managed cloud service, explore [Marqo Cloud](https://cloud.marqo.ai).

### Documentation

Comprehensive Marqo documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

### Warning

Do not run other applications on Marqo's Vespa cluster, as Marqo automatically manages and adjusts the cluster settings.

### Contributors

Marqo is a community project.  See [CONTRIBUTING.md](./CONTRIBUTING.md) for information on contributing.

### Dev Setup

1.  Create a virtual env: `python -m venv ./venv`.
2.  Activate the virtual environment: `source ./venv/bin/activate`.
3.  Install requirements: `pip install -r requirements.txt`.
4.  Run tests: Run `tox`.
5.  If dependencies change, delete the `.tox` directory and rerun.

### Merge Instructions

1.  Run the full test suite (using `tox`).
2.  Create a pull request with a linked GitHub issue.

### Support

*   [Discourse forum](https://community.marqo.ai)
*   [Slack community](https://bit.ly/marqo-community-slack)