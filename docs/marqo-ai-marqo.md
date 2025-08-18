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

## Marqo: The Open-Source Vector Search Engine for Text and Images

**Marqo empowers developers to build cutting-edge search applications with ease, offering end-to-end vector search capabilities for both text and images.** ([View on GitHub](https://github.com/marqo-ai/marqo))

### 🚀 Key Features

*   **State-of-the-Art Embeddings:**
    *   Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU acceleration.
*   **Blazing Fast Performance:**
    *   In-memory HNSW indexes for lightning-fast search speeds.
    *   Scale to hundreds of millions of documents with horizontal index sharding.
    *   Asynchronous and non-blocking data upload and search.
*   **Documents-In, Documents-Out:**
    *   Vector generation, storage, and retrieval are handled out-of-the-box.
    *   Build search, entity resolution, and data exploration applications with text and images.
    *   Create complex semantic queries using weighted search terms.
    *   Filter results using Marqo’s query DSL.
    *   Store unstructured data and metadata together using various datatypes.
*   **Seamless Integrations:**
    *   Integrations with Haystack, Griptape, Langchain and Hamilton to supercharge your LLM and NLP workflows.
*   **Managed Cloud Option:**
    *   Optimized deployment of Marqo with low latency.
    *   Scale inference with a click.
    *   High availability.
    *   24/7 Support
    *   Access control.
    *   [Learn more](https://www.marqo.ai/cloud).

### 💡 Why Use Marqo?

Marqo simplifies vector search by providing a comprehensive solution, eliminating the need to manage separate components for ML model deployment, preprocessing, and input transformations. It's "documents in, documents out" approach allows you to build advanced search applications with minimal effort.

### 💻 Quick Start

Get up and running with Marqo in minutes:

1.  **Install Docker:**  Follow the instructions on the [Docker Official website](https://docs.docker.com/get-docker/), and ensure Docker has at least 8GB memory and 50GB storage allocated.
2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```
3.  **Install the Marqo client:**
    ```bash
    pip install marqo
    ```
4.  **Index and Search:**
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

    **Note:** If your `marqo` container is getting killed, increase the Docker memory allocation (at least 6GB, 8GB recommended).

### 📚 Learn More

*   [Quick Start Guide](#getting-started)
*   [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)
*   [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)
*   [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering)
*   [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs)
*   [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)
*   [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)
*   [Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)
*   [Marqo for E-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)
*   [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)
*   [Core Features](#-core-features)

### ⚙️ Additional Operations

*   [Get Document](#get-document)
*   [Get Index Stats](#get-index-stats)
*   [Lexical Search](#lexical-search)
*   [Multimodal and Cross Modal Search](#multi-modal-and-cross-modal-search)
*   [Searching Using an Image](#searching-using-an-image)
*   [Searching using weights in queries](#searching-using-weights-in-queries)
*   [Creating and searching indexes with multimodal combination fields](#creating-and-searching-indexes-with-multimodal-combination-fields)
*   [Delete Documents](#delete-documents)
*   [Delete Index](#delete-index)

### 🚀 Production Deployment

*   [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   [Marqo Cloud](https://cloud.marqo.ai)

### 📖 Documentation

Explore the full Marqo documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

### 🤝 Contributing

Marqo is a community-driven project!  Learn how to contribute [here](./CONTRIBUTING.md).

### 💬 Support

*   [Discourse Forum](https://community.marqo.ai)
*   [Slack Community](https://bit.ly/marqo-community-slack)

---

### 🛠️ Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`

**Important:** Delete the `.tox` directory and rerun `tox` if you update dependencies.

### 📜 Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with an attached GitHub issue.