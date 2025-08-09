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

**Marqo is a powerful, open-source vector search engine that simplifies the process of indexing, searching, and retrieving both text and images with a single, easy-to-use API.**  [Explore the Marqo GitHub Repository](https://github.com/marqo-ai/marqo)

### **🚀 Key Features**

*   **Seamless Vector Generation:** Eliminate the need to bring your own embeddings; Marqo handles vector generation, storage, and retrieval out-of-the-box.
*   **Multimodal Search:** Effortlessly search and retrieve both text and images using cutting-edge CLIP models.
*   **Blazing Fast Performance:** Experience lightning-fast search speeds with in-memory HNSW indexes and horizontal index sharding.
*   **Documents-in, Documents-Out:** Simplify your workflow with Marqo's document-centric approach, handling preprocessing, embedding, and metadata storage.
*   **State-of-the-Art Embeddings:** Leverage the latest machine learning models from Hugging Face, OpenAI, and more, with CPU and GPU support.
*   **Managed Cloud Option:**  Take advantage of Marqo Cloud for low-latency deployment, scalability, and 24/7 support.

### Quick Start

Follow these steps to get started with Marqo:

1.  **Install Docker:** Ensure you have Docker installed and allocated at least 8GB of memory and 50GB of storage.  See [Docker Official website](https://docs.docker.com/get-docker/)
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
4.  **Index and Search!**

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

*   **🤖 State-of-the-art embeddings:** Utilize cutting-edge machine learning models from various providers, with pre-configured models or the ability to bring your own.  Supports both CPU and GPU acceleration.
*   **⚡ Performance:** Benefit from high-speed search utilizing in-memory HNSW indexes, capable of scaling to indexes with hundreds of millions of documents with horizontal index sharding and asynchronous data upload and search.
*   **🌌 Documents-in-documents-out:** Index and search by both text and images out of the box. Supports complex search queries by combining terms and filtering results.
*   **🍱 Managed cloud:** Easily deploy and scale with Marqo Cloud, offering optimized deployment, scaling, high availability, and 24/7 support.

### Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack):** Use Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape):** Access scalable search within LLM-based agents.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain):** Integrate Marqo into LangChain applications for vector search.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Leverage Marqo within Hamilton LLM applications.

### Learn More About Marqo

*   📗 [Quick start](#getting-started): Get your first application up and running in minutes.
*   🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization): Advanced image search with Marqo.
*   📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code): Building a multilingual database.
*   🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering): Using Marqo as a knowledge base with GPT.
*   🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs): Combine stable diffusion with semantic search.
*   🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing): Preprocess audio for Q&A.
*   🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo): Building advanced image search for content removal.
*   ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud): Set up and start with Marqo Cloud.
*   👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md): E-commerce demo with ReactJS and Flask.
*   🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo): Build a chatbot with Marqo and OpenAI's ChatGPT API.
*   🦾 [Features](#-Core-Features): Marqo's core features.

### Detailed Usage Examples

*   **Get Document:** Retrieve documents by ID.
    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```
*   **Get Index Stats:** Get index information.
    ```python
    results = mq.index("my-first-index").get_stats()
    ```
*   **Lexical Search:** Perform keyword search.
    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```
*   **Multimodal and Cross-Modal Search:** Index and search images and text. Create an index and use URLs from the internet or local disk:
    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)

    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus...",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    results = mq.index("my-multimodal-index").search('animal')
    ```
*   **Search with Image:**
    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```
*   **Weighted Queries:** Use dictionaries with weights for advanced queries.
    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    results = mq.index("my-weighted-query-index").search(q=query)
    ```
*   **Multimodal Combination Fields:**  Combine text and images in one field:
    ```python
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7
            }
        }
    }
    ```
*   **Delete Documents:**
    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```
*   **Delete Index:**
    ```python
    results = mq.index("my-first-index").delete()
    ```

### Production Deployment

*   **Kubernetes:** Deploy Marqo using Kubernetes templates ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)).
*   **Marqo Cloud:**  For a fully managed experience, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

### Important Notes

*   Do not run other applications on Marqo's Vespa cluster as Marqo manages and adjusts the settings.

### Contributing

Marqo thrives on community contributions!  Read the [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

### Development Setup

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the environment: `source ./venv/bin/activate`.
3.  Install requirements: `pip install -r requirements.txt`.
4.  Run tests using tox: `tox`.
5.  If dependencies change, delete `.tox` and rerun tox.

### Merge Instructions

1.  Run the full test suite (`tox`).
2.  Create a pull request with a linked GitHub issue.

### Support

*   Join the conversation and share your creations on our [Discourse forum](https://community.marqo.ai).
*   Connect with the community on our [Slack community](https://bit.ly/marqo-community-slack).