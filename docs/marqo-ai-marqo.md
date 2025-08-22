<p align="center">
<img src="https://github.com/marqo-ai/public-assets/blob/main/marqowbackground.png" width="50%" height="40%">
</p>

<p align="center">
<b><a href="https://www.marqo.ai">Website</a> | <a href="https://docs.marqo.ai">Documentation</a> | <a href="https://demo.marqo.ai">Demos</a> | <a href="https://www.marqo.ai/cloud">Marqo Cloud</a>
</b>
</p>

<p align="center">
<a href="https://opensource.org/licenses/Apache-2.0"><img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg"></a>
<a href="https://pypi.org/project/marqo/"><img src="https://img.shields.io/pypi/v/marqo?label=PyPI"></a>
<a href="https://github.com/marqo-ai/marqo/actions/workflows/unit_test_200gb_CI.yml"><img src="https://img.shields.io/github/actions/workflow/status/marqo-ai/marqo/unit_test_200gb_CI.yml?branch=mainline"></a>

# Marqo: The Open-Source Vector Search Engine for Text and Images

**Effortlessly build cutting-edge search applications with Marqo, the all-in-one vector search engine.**  [Explore the original repository](https://github.com/marqo-ai/marqo).

## Key Features

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU.

*   **⚡ High Performance:**
    *   Leverage in-memory HNSW indexes for blazing-fast search speeds.
    *   Scale to indexes with hundreds of millions of documents using horizontal index sharding.
    *   Benefit from asynchronous and non-blocking data upload and search operations.

*   **🌌 Documents-In-Documents-Out:**
    *   Vector generation, storage, and retrieval are handled seamlessly.
    *   Create search, entity resolution, and data exploration applications using your text and images.
    *   Craft complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo’s query DSL.
    *   Store structured metadata and unstructured data together using a range of supported datatypes.

*   **🍱 Managed Cloud Offering:**
    *   Low-latency deployments are optimized for performance.
    *   Easily scale inference with a single click.
    *   Offers high availability and 24/7 support.
    *   Includes access control for secure management.
    *   Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Use Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Integrate Marqo for scalable search within your LLM-based agents.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Implement Marqo for vector search components in LangChain applications.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Leverage Marqo for Hamilton LLM applications.

## Quick Start

Get started with Marqo using Docker and Python:

1.  **Install Docker**: Ensure Docker is installed and has at least 8GB memory and 50GB storage. (See [Docker Official website](https://docs.docker.com/get-docker/).)
2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
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

For more details, including a comprehensive list of available models, refer to the [Models Reference](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) in the Marqo documentation.

## Learn More

Explore these resources to expand your Marqo knowledge:

*   📗 [Quick Start](#getting-started): Build your first application in minutes.
*   🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization): Advanced image search techniques.
*   📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code): Building a multilingual database.
*   🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering): Enhancing GPT with Marqo.
*   🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs): Semantic search with Stable Diffusion.
*   🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing): Preprocessing audio for Q&A.
*   🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo): Image content moderation.
*   ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud): Setup and deployment guide.
*   👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md): Build an e-commerce web application.
*   🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo): Build a chatbot application.
*   🦾 [Core Features](#-Core-Features): Explore Marqo's core features.

## Basic Operations

*   **Get Document:** Retrieve a document by ID.
    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```

*   **Get Index Stats:** Retrieve index information.
    ```python
    results = mq.index("my-first-index").get_stats()
    ```

*   **Lexical Search:** Perform keyword-based search.
    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```

*   **Multimodal and Cross-Modal Search:**
    *   Configure your index for images:
        ```python
        settings = {
            "treat_urls_and_pointers_as_images":True,
            "model":"ViT-L/14"
        }
        response = mq.create_index("my-multimodal-index", **settings)
        ```
    *   Add images within documents.
        ```python
        response = mq.index("my-multimodal-index").add_documents([{
            "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
            "Description": "The hippopotamus...",
            "_id": "hippo-facts"
        }], tensor_fields=["My_Image"])
        ```
    *   Search by text or image URL.
        ```python
        results = mq.index("my-multimodal-index").search('animal')
        results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
        ```

*   **Weighted Queries:** Refine search with query weights.
    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```

*   **Multimodal Combination Fields:** Combine text and images.

    ```python
    settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}
    mq.create_index("my-first-multimodal-index", **settings)
    mq.index("my-first-multimodal-index").add_documents(
        [
            {
                "Title": "Flying Plane",
                "caption": "An image of a passenger plane flying in front of the moon.",
                "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
            },
        ],
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
    )
    ```

*   **Delete Documents:** Delete documents by ID.
    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```

*   **Delete Index:** Delete an index.
    ```python
    results = mq.index("my-first-index").delete()
    ```

## Production Deployment

*   **Kubernetes:** Deploy Marqo with Kubernetes templates: [marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** For a fully managed service, sign up at [cloud.marqo.ai](https://cloud.marqo.ai).

## Important Notes

*   Do not run other applications on Marqo's Vespa cluster as Marqo automatically changes and adapts the settings on the cluster.

## Contributing

We welcome contributions!  See the [CONTRIBUTING.md](./CONTRIBUTING.md) file for details.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`
5.  Update dependencies by deleting the `.tox` directory and rerunning `tox`.

## Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.

## Support

*   [Discourse forum](https://community.marqo.ai)
*   [Slack community](https://bit.ly/marqo-community-slack)