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

## Marqo: The Open-Source, End-to-End Vector Search Engine for Text and Images

Marqo simplifies vector search, providing an all-in-one solution for generating, storing, and retrieving vectors from both text and images through a single API. **[See the original repo](https://github.com/marqo-ai/marqo)**.

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Utilize cutting-edge machine learning models from PyTorch, Hugging Face, and OpenAI.  Easily use pre-configured models or bring your own, with CPU and GPU support.
*   **⚡ High Performance:** Benefit from in-memory HNSW indexes for blazing-fast search speeds and the ability to scale to hundreds of millions of documents. Achieve asynchronous, non-blocking data upload and search.
*   **🌌 Documents-In-Documents-Out:**  Vector generation, storage, and retrieval are provided out of the box. Build search, entity resolution, and data exploration applications with your text and images. Create complex semantic queries using weighted search terms and filter results with Marqo's query DSL. Support a range of datatypes including bools, ints and keywords.
*   **🍱 Managed Cloud Option:**  Access a low-latency, optimized Marqo deployment with scalable inference, high availability, and 24/7 support.  Learn more [here](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Search:** Easily search across text and images, allowing you to create richer, more relevant search experiences.

**Why Marqo?**

Marqo goes beyond the capabilities of a standard vector database by integrating machine learning model management, preprocessing, and the ability to modify search behavior without retraining.  This holistic approach allows developers to build powerful vector search capabilities into their applications with minimal effort.

**Quick Start**

1.  **Prerequisites:** Marqo requires Docker. Install Docker from the [official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage.
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
4.  **Index and Search:** Here's a minimal example using Python:

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

**Core Functionality:**

*   `create_index()`: Create a new index, optionally specifying a model.
*   `add_documents()`: Add documents for indexing.
*   `search()`: Perform a vector search.

**Advanced Operations:**

*   **Get Document:** Retrieve a document by ID.
*   **Get Index Stats:** Retrieve information about an index.
*   **Lexical Search:** Perform a keyword search.
*   **Multimodal and Cross-Modal Search:** Combine text and image search.
*   **Searching using weights in queries** Create more advanced queries consisting of multiple components with weightings towards or against them
*   **Creating and searching indexes with multimodal combination fields** Marqo lets you have indexes with multimodal combination fields.

**Integrations:**

Marqo integrates with popular AI and data processing frameworks.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

**Learn More:**

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
*   **[Running Marqo Open Source in Production](https://github.com/marqo-ai/marqo-on-kubernetes)**
*   **[Marqo Cloud](https://cloud.marqo.ai)**

**Community & Support:**

*   Join our [Discourse forum](https://community.marqo.ai) to ask questions and share ideas.
*   Join our [Slack community](https://bit.ly/marqo-community-slack) to chat with others.

**Contributing:**

We welcome your contributions! Please see the [CONTRIBUTING.md](./CONTRIBUTING.md) file for details.

**Dev Setup:**

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the virtual environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`
5.  Update dependencies: Delete the `.tox` directory and rerun `tox`.

**Merge Instructions:**

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.

**Important Note:**  Avoid running other applications on Marqo's Vespa cluster, as Marqo dynamically adjusts cluster settings.