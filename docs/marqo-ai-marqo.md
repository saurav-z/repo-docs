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

Marqo is an open-source, end-to-end vector search engine that makes it easy to search both text and images with a single API, offering vector generation, storage, and retrieval out of the box.  ([See the original repo](https://github.com/marqo-ai/marqo)).

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more. Choose from pre-configured models or bring your own, with support for CPU and GPU.
*   **⚡ High Performance:** Experience cutting-edge search speeds with embeddings stored in in-memory HNSW indexes, scale to hundreds of millions of documents, and benefit from async, non-blocking data upload and search.
*   **🌌 Documents-in-Documents-out:** Simplify vector search with out-of-the-box vector generation, storage, and retrieval. Build semantic search applications, entity resolution, and data exploration using your text and images.
*   **🍱 Managed Cloud Option:** Get low-latency optimized deployment, easy inference scaling, high availability, 24/7 support, and access control with Marqo Cloud. Learn more [here](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Search**: Search text and images with single API calls using CLIP models
*   **✨ Multimodal Combination Fields**: Combine text and images into one field for richer, more efficient search.

## Integrations

Marqo integrates with popular AI and data processing frameworks, enabling you to enhance your applications.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More about Marqo

| Resources | Description |
|---|---|
| 📗 [Quick Start](#Getting-started)| Get started with Marqo in under 5 minutes. |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo. |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base. |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT. |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content. |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo|
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.|
| 🦾 [Features](#-Core-Features) | Marqo's core features. |


## Getting Started

1.  **Prerequisites:** Marqo requires Docker. Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure that docker has at least 8GB memory and 50GB storage.

2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```
    *   **Note:** If your `marqo` container is getting killed, increase Docker's memory limit (at least 6GB, 8GB recommended) in Docker settings.

3.  **Install the Marqo Client:**
    ```bash
    pip install marqo
    ```

4.  **Index and Search:**

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2") # optional model

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

    *   `mq` is the Marqo client.
    *   `create_index()`: Creates a new index (optionally with a specified model, like `hf/e5-base-v2`).
    *   `add_documents()`: Indexes your data.  `tensor_fields` specify fields for vector indexing.
    *   You can set a document ID using `_id`. Otherwise Marqo will generate one.

    **Results Example:**
    ```python
    import pprint
    pprint.pprint(results)
    ```
    The results will be displayed showing the most relevant documents.

## Other Basic Operations

*   **Get Document:** `mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** `mq.index("my-first-index").get_stats()`
*   **Lexical Search:** `mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal and Cross-Modal Search:** Use CLIP models for image and text search. Create an index with settings:  `settings = {"treat_urls_and_pointers_as_images":True, "model":"ViT-L/14"}`. Add images with URLs, then search using text queries, e.g., `mq.index("my-multimodal-index").search('animal')`.
*   **Searching Using an Image:** Search with image URLs. `results = mq.index("my-multimodal-index").search('https://...')`
*   **Searching Using Weights in Queries:** Construct advanced queries using weighted terms for refined search results.
*   **Creating and Searching Indexes with Multimodal Combination Fields:** Create indexes with multimodal combination fields to combine text and images. This allows a single vector representation and the ability to search using queries with weighted components.
*   **Delete Documents:** `results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])`
*   **Delete Index:** `results = mq.index("my-first-index").delete()`

## Running Marqo in Production

*   **Kubernetes:** Use our Kubernetes templates for deployment on your cloud provider. [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** For a fully managed service, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Find detailed documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Warning

Do not run other applications on Marqo's Vespa cluster, as Marqo manages and adapts its settings.

## Contributing

Marqo is a community-driven project.  Contributions are welcome!  See [CONTRIBUTING.md](./CONTRIBUTING.md) for details.

## Dev Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests: `tox` (from this directory)
5.  If you update dependencies, delete `.tox` and rerun `tox`.

## Merge Instructions

1.  Run the full test suite (`tox`).
2.  Create a pull request with an attached GitHub issue.

## Support

*   [Discourse forum](https://community.marqo.ai) - for questions and sharing.
*   [Slack community](https://bit.ly/marqo-community-slack) - for chatting with other members.