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

**Marqo is an end-to-end vector search engine that simplifies building semantic search applications for both text and images, handling vector generation, storage, and retrieval with a single API.** ([View on GitHub](https://github.com/marqo-ai/marqo))

### Key Features:

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
*   **⚡ High Performance:** Experience cutting-edge search speeds with embeddings stored in in-memory HNSW indexes.
*   **🌌 Documents-in-Documents-Out:** Simplify your workflow with out-of-the-box vector generation, storage, and retrieval.
*   **🍱 Managed Cloud Option:** Benefit from low-latency, optimized deployments with scalable inference, high availability, and 24/7 support. (Learn more at [Marqo Cloud](https://www.marqo.ai/cloud)).
*   **🖼️ Multimodal Search:** Easily search text and images using a single API, including CLIP models.

### Quick Start

Get started with Marqo in a few steps:

1.  **Install Docker:**  Follow the instructions on the [Docker Official website](https://docs.docker.com/get-docker/), ensuring at least 8GB of memory and 50GB of storage.

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

### Core Features
See above
### Integrations

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

### Learn More

*   📗 [Quick start](#Getting-started)| Build your first application with Marqo in under 5 minutes.
*   🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo.
*   📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo.
*   🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.
*   🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.
*   🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content.
*   ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo|
*   👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
*   🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.|
*   🦾 [Features](#-Core-Features) | Marqo's core features.

### Getting Started
See above

### Other Basic Operations

*   **Get Document:**  Retrieve a document by ID.
*   **Get Index Stats:** Get information about an index.
*   **Lexical Search:** Perform keyword search.
*   **Multimodal and Cross-Modal Search:**  Search across text and images.
*   **Searching with Weights in Queries:** Customize your search queries using weighted terms.
*   **Creating and Searching Indexes with Multimodal Combination Fields:** Combine and search across text and images in a single field.
*   **Delete Documents:** Delete specific documents.
*   **Delete Index:** Delete an index.

### Running Marqo in Production

Marqo provides Kubernetes templates for deployment on your preferred cloud provider ([marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)). For a fully managed cloud service, consider [Marqo Cloud](https://cloud.marqo.ai).

### Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

### Warning

Avoid running other applications on Marqo's Vespa cluster as Marqo automatically adjusts its settings.

### Contributing

We welcome contributions! Please review the [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.

### Dev Setup
See above
### Merge Instructions
See above
### Support

*   Join the conversation on our [Discourse forum](https://community.marqo.ai).
*   Connect with the community on our [Slack channel](https://bit.ly/marqo-community-slack).