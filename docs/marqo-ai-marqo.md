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

**Marqo empowers developers to build advanced search applications with an all-in-one vector search engine, simplifying the process of vector generation, storage, and retrieval.**  Find out more on the [original repo](https://github.com/marqo-ai/marqo).

### 🔑 Key Features:

*   **State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU for optimal performance.
*   **High Performance & Scalability:**
    *   Utilizes in-memory HNSW indexes for blazing-fast search speeds.
    *   Scales to handle indexes with hundreds of millions of documents through horizontal index sharding.
    *   Offers async and non-blocking data upload and search capabilities.
*   **Documents-in, Documents-Out:**
    *   Simplifies vector search by providing vector generation, storage, and retrieval out of the box.
    *   Enables building search, entity resolution, and data exploration applications using your text and images.
    *   Supports complex semantic queries using weighted search terms and a query DSL for filtering.
    *   Allows storage of unstructured data and semi-structured metadata with a range of supported datatypes (bools, ints, keywords, etc).
*   **Managed Cloud Option:**
    *   Offers low-latency optimized deployment of Marqo.
    *   Scale inference easily.
    *   Provides high availability, 24/7 support, and access control.
    *   Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).

### 🚀 Quick Start

Get up and running with Marqo in minutes using Docker and Python:

1.  **Install Docker:**  Follow the [Docker Official website](https://docs.docker.com/get-docker/) instructions, ensuring at least 8GB memory and 50GB storage are allocated.

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

### 📚 Learn More

*   **[Quick start](#quick-start):** Build your first application with Marqo in under 5 minutes.
*   **[Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization):** Building advanced image search with Marqo.
*   **[Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code):** Building a multilingual database in Marqo.
*   **[Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering):** Making GPT a subject matter expert by using Marqo as a knowledge base.
*   **[ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs):** Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.
*   **[Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing):** Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   **[Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo):** Building advanced image search with Marqo to find and remove content.
*   **[Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud):** Go through how to get set up and running with Marqo Cloud.
*   **[Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md):** A web application with frontend and backend using Python, Flask, ReactJS, and Typescript.
*   **[Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo):** Build a chat bot application using Marqo and OpenAI's ChatGPT API.
*   **[Core Features](#-core-features):** Marqo's core features.

### 🤝 Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack):** Use Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape):** Access scalable search with your own data for LLM-based agents.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain):** Leverage open source or custom fine tuned models through Marqo for LangChain applications.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Leverage open source or custom fine tuned models through Marqo for Hamilton LLM applications.

### ⚙️ Advanced Usage

*   **Get document:**

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*   **Get index stats:**

```python
results = mq.index("my-first-index").get_stats()
```

*   **Lexical search:**

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

*   **Multi modal and cross modal search:** Create an index with a CLIP configuration for image and text search.  Images can be added within documents using urls.

```python
settings = {
    "treat_urls_and_pointers_as_images":True,
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```
*   **Searching using an image:** Search using an image by providing an image link:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

*   **Searching using weights in queries:** Use queries as dictionaries with weighted terms.

```python
query = {
    "I need to buy a communications device, what should I get?": 1.1,
    "The device should work like an intelligent computer.": 1.0,
}

results = mq.index("my-weighted-query-index").search(q=query)
```

*   **Creating and searching indexes with multimodal combination fields:** Combine text and images into one field.

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

*   **Delete documents:**

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

*   **Delete index:**

```python
results = mq.index("my-first-index").delete()
```
### 🚀 Running Marqo in Production

Marqo supports Kubernetes templates and a fully managed cloud service:

*   **Kubernetes:** Deploy on your preferred cloud provider using our Kubernetes templates: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** Sign up for a fully managed cloud service here: [https://cloud.marqo.ai](https://cloud.marqo.ai).

### 📖 Documentation

Comprehensive documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/).

### ⚠️ Important Notes

*   Do not run other applications on Marqo's Vespa cluster, as Marqo automatically manages its settings.

### 🙏 Contributing

Marqo is a community project.  Contributions are welcome! See [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

### 💬 Support

*   Join the [Discourse forum](https://community.marqo.ai) to ask questions and share your work.
*   Connect with the community on our [Slack community](https://bit.ly/marqo-community-slack).

### 💻 Development Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`
5.  Update dependencies:  Delete `.tox` and rerun `tox` after updating dependencies.

### 🤝 Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.