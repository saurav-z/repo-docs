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

**Supercharge your applications with Marqo, an end-to-end vector search engine that simplifies vector generation, storage, and retrieval for both text and images.** Built for developers, Marqo eliminates the need to manage embeddings and ML models separately, streamlining the process of integrating powerful search capabilities.  [Explore Marqo on GitHub](https://github.com/marqo-ai/marqo).

### Key Features

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Leverage pre-configured models or bring your own custom models.
    *   Enjoy both CPU and GPU support for optimal performance.
*   **⚡ High Performance:**
    *   Benefit from embeddings stored in in-memory HNSW indexes for rapid search speeds.
    *   Scale effortlessly to handle indexes containing hundreds of millions of documents with horizontal index sharding.
    *   Experience async and non-blocking data upload and search operations.
*   **🌌 Documents-in-Documents-Out:**
    *   Vector generation, storage, and retrieval are provided out of the box.
    *   Easily build search, entity resolution, and data exploration applications using both text and images.
    *   Construct complex semantic queries by combining weighted search terms.
    *   Refine search results using Marqo's query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, with support for datatypes like bools, ints and keywords.
*   **🍱 Managed Cloud Option:**
    *   Access low-latency, optimized Marqo deployments.
    *   Scale inference seamlessly with a click of a button.
    *   Enjoy high availability and 24/7 support.
    *   Benefit from access control for enhanced security.
    *   Learn more [here](https://www.marqo.ai/cloud).

### Integrations

Marqo integrates seamlessly with leading AI and data processing frameworks, with more integrations continuously being added.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**:  Use Marqo as your Document Store for building Haystack pipelines such as retrieval-augmentation, question answering, and document search.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**:  Integrate with MarqoVectorStoreDriver for scalable search within LLM-based agent applications.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Utilize Marqo for vector search components within LangChain applications, including Retrieval QA and Conversational Retrieval QA.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Leverage Marqo for Hamilton LLM applications.

### Learn More

Explore these resources to get started and dive deeper into Marqo's capabilities:

| Resource                                                                                                                                       | Description                                                                                            |
| ------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------ |
| 📗 [Quick start](#getting-started)                                                                                                             | Build your first application with Marqo in minutes.                                                  |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)                        | Building advanced image search with Marqo.                                                            |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)                     | Building a multilingual database in Marqo.                                                            |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.                                  |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorize 100k images of hot dogs.   |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                                                                          | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                   |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)                      | Building advanced image search with Marqo to find and remove content.                                 |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)                                                | Start building your first application on Marqo Cloud.                                                   |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)                          | Build a web application using Python, Flask, ReactJS, and Typescript for e-commerce.                     |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)                                                 | Build a chatbot application using Marqo and OpenAI's ChatGPT API.                                       |
| 🦾 [Features](#-Core-Features)                                                                                                                   | Explore Marqo's core features.                                                                          |

### Getting Started

Follow these steps to quickly set up and run Marqo:

1.  **Install Docker:** Go to the [Docker Official website](https://docs.docker.com/get-docker/) and install Docker. Ensure that Docker is allocated at least 8GB of memory and 50GB of storage.
2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If the `marqo` container is being killed due to a lack of memory, increase the memory limit in Docker settings to at least 6GB (8GB is recommended).*
3.  **Install the Marqo client:**
    ```bash
    pip install marqo
    ```
4.  **Start Indexing and Searching!** Below is a simple example:
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
    *   `mq` is the client, wrapping the `marqo` API.
    *   `create_index()` creates a new index, where you can specify the model to use (e.g., `hf/e5-base-v2`).
    *   `add_documents()` indexes documents from a list of Python dictionaries, where `tensor_fields` indicate the fields that will be indexed as vector collections.
    *   You can set a document's ID with the `_id` field. Otherwise, Marqo generates one.
5.  **View Results:**
    ```python
    # Print the results:
    import pprint
    pprint.pprint(results)
    ```
    *   Each hit is a matched document, ordered by relevance.
    *   `_highlights` highlights the matching section of the document.
    *   `_score` reflects the match relevance.

### Other Basic Operations

*   **Get Document:** Retrieve a document by its ID.
    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```
*   **Get Index Stats:** Retrieve index information.
    ```python
    results = mq.index("my-first-index").get_stats()
    ```
*   **Lexical Search:** Perform keyword searches.
    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```
*   **Multi-Modal and Cross-Modal Search:**  Use CLIP models from Hugging Face to enable image and text search.  Create an index with a CLIP configuration:
    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```
    Add images within documents using URLs or local paths:
    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus...",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```
    Then, search the image field using text:
    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```
    or search using an image itself:
    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```
*   **Weighted Queries:** Construct advanced queries with weighted components:
    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```
*   **Multimodal Combination Fields:** Combine text and images within a single field, with per-document weighting:
    ```python
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7
            }
        }
    },
    tensor_fields=["captioned_image"]
    ```
*   **Delete Documents:** Delete documents by ID.
    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```
*   **Delete Index:** Delete an index.
    ```python
    results = mq.index("my-first-index").delete()
    ```

### Production Deployment

*   **Kubernetes:**  Marqo provides Kubernetes templates for deploying clusters, including replicas, storage sharding, and inference nodes.  Find the repo here:  [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** For a fully managed cloud service, sign up at:  [https://cloud.marqo.ai](https://cloud.marqo.ai)

### Documentation

Comprehensive documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/)

### Important Note

Avoid running other applications on Marqo's Vespa cluster. Marqo automatically configures and adapts the cluster's settings.

### Contributing

Marqo thrives on community contributions.  Read [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

### Development Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate the environment:  `source ./venv/bin/activate`
3.  Install requirements:  `pip install -r requirements.txt`
4.  Run tests:  `tox`  (within the project directory)
5.  If you update dependencies, delete the `.tox` directory and rerun `tox`.

### Merge Instructions

1.  Run the full test suite (using `tox`).
2.  Create a pull request, linked to a GitHub issue.

### Support

*   Join the [Discourse forum](https://community.marqo.ai) to ask questions and share your projects.
*   Connect with the community on our [Slack community](https://bit.ly/marqo-community-slack)