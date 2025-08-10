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

**Marqo simplifies vector search with a single API, handling vector generation, storage, and retrieval for both text and images.**  [Explore the Marqo repository](https://github.com/marqo-ai/marqo).

### Key Features:

*   **🤖 State-of-the-art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, and more, with both CPU and GPU support.
*   **⚡ High Performance:** Experience cutting-edge search speeds with in-memory HNSW indexes and scale to millions of documents with horizontal sharding.
*   **🌌 Documents-in-Documents-Out:** Simplify your workflow with out-of-the-box vector generation, storage, and retrieval for text and images.
*   **🍱 Managed Cloud Option:** Access a fully managed cloud service with low latency, scalable inference, high availability, and 24/7 support. Learn more at [Marqo Cloud](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Capabilities:** Easily search both text and image data with CLIP models.
*   **💪 Flexible Querying:** Build complex semantic queries, including weighted searches and filtering with Marqo's query DSL.

### Why Choose Marqo?

Unlike traditional vector databases, Marqo is a complete vector search engine. It streamlines the process by handling:

*   **Simplified Setup:** Easily build vector search into your applications with minimal effort.
*   **Automated Processes:** Preprocessing, embedding generation, metadata storage, and model deployment are all managed by Marqo.
*   **End-to-End Solution:** Marqo provides a complete solution allowing you to focus on your core application.

### Getting Started

Follow these steps to quickly start with Marqo:

1.  **Install Docker:** Ensure Docker is installed.  See the [Docker Official website](https://docs.docker.com/get-docker/) for installation instructions. Allocate at least 8GB memory and 50GB storage.

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

4.  **Index and Search:**  See the example below.

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
### Core Features
*   **🤖 State of the art embeddings**
    - Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more.
    - Start with a pre-configured model or bring your own.
    - CPU and GPU support.

*   **⚡ Performance**
    - Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
    - Scale to hundred-million document indexes with horizontal index sharding.
    - Async and non-blocking data upload and search.

*   **🌌 Documents-in-documents-out**
    - Vector generation, storage, and retrieval are provided out of the box.
    - Build search, entity resolution, and data exploration application with using your text and images.
    - Build complex semantic queries by combining weighted search terms.
    - Filter search results using Marqo’s query DSL.
    - Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

*   **🍱 Managed cloud**
    - Low latency optimised deployment of Marqo.
    - Scale inference at the click of a button.
    - High availability.
    - 24/7 support.
    - Access control.
    - Learn more [here](https://www.marqo.ai/cloud).

### Integrations

Marqo integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

### Learn More

Explore these resources to deepen your understanding of Marqo:

*   📗 [Quick Start](#getting-started) - Get up and running with Marqo in minutes.
*   🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) - Advanced image search techniques.
*   📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) - Building multilingual databases.
*   🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) - Enhancing GPT with Marqo.
*   🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) - Combining Stable Diffusion with semantic search.
*   🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) - Preprocessing audio data.
*   🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) - Content moderation with Marqo.
*   ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) - Get started with Marqo Cloud.
*   👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) - E-commerce demo.
*   🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) - Chatbot demo.
*   🦾 [Core Features](#core-features) - A summary of Marqo's key features.

### Advanced Operations

*   **Get Document:**

    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```

*   **Get Index Stats:**

    ```python
    results = mq.index("my-first-index").get_stats()
    ```

*   **Lexical Search:**

    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```

*   **Multimodal Search:** Create indexes with CLIP models:

    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```

    Add images via URLs:

    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus...",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```

    Search using an image:
    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```
    Or search with text:
    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```

*   **Weighted Queries:**

    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```
    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.0,
        "The device should work like an intelligent computer.": -0.3,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```

*   **Multimodal Combination Fields:** Combine text and image search.

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

*   **Delete Documents:**

    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```

*   **Delete Index:**

    ```python
    results = mq.index("my-first-index").delete()
    ```

### Running in Production

*   **Kubernetes:** Deploy Marqo using provided Kubernetes templates: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed experience, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

### Important Note

Avoid running other applications on Marqo's Vespa cluster, as Marqo automatically adjusts cluster settings.

### Contributing

Marqo is a community project.  See [CONTRIBUTING.md](./CONTRIBUTING.md) to learn how to contribute.

### Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests:  `tox` (after `cd` into the directory)
5.  Update dependencies and rerun `tox` by deleting the `.tox` directory.

### Merge Instructions:

1.  Run the full test suite using `tox`.
2.  Create a pull request with a linked GitHub issue.

### Support

*   Join the discussion on our [Discourse forum](https://community.marqo.ai).
*   Join the [Slack community](https://bit.ly/marqo-community-slack).