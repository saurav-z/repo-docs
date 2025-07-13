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

**Marqo empowers developers to build powerful vector search applications with an easy-to-use, end-to-end solution for both text and image data.** ([View the original repository](https://github.com/marqo-ai/marqo))

### Key Features:

*   **⚡ Fast & Efficient:** Utilize in-memory HNSW indexes for blazing-fast search speeds and scale to hundreds of millions of documents.
*   **🤖 State-of-the-Art Embeddings:** Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more, with CPU and GPU support.
*   **🌌 Documents-in, Documents-out:** Simplify your workflow with built-in vector generation, storage, and retrieval for text and images.
*   **🍱 Managed Cloud Option:** Benefit from a fully managed cloud service with low latency, scalability, high availability, 24/7 support, and access control.
*   **🌐 Multimodal Search:** Easily index and search both text and image data with the power of CLIP models.
*   **📚 Integrations:** Seamlessly integrate with popular AI frameworks such as Haystack, Griptape, Langchain, and Hamilton.
*   **🎨 Flexible Querying:** Build complex semantic queries with weighted search terms and filter results using Marqo's query DSL.

### Why Choose Marqo?

Unlike traditional vector databases, Marqo provides a complete vector search solution. You no longer need to manage machine learning model deployment, preprocessing, and input transformations separately. Marqo is designed for ease of use with "documents in, documents out" functionality. Quickly build semantic search, entity resolution, and data exploration applications with Marqo's intuitive API.

### Quick Start Guide

Get started with Marqo in a few simple steps:

1.  **Install Docker:**  If you do not have Docker installed, visit the [Docker Official website](https://docs.docker.com/get-docker/) for instructions. Ensure Docker has at least 8GB of memory and 50GB of storage allocated.
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
4.  **Start Indexing and Searching:**  Here's a minimal example:
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

### Other Basic Operations

*   **Get Document:** `result = mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** `results = mq.index("my-first-index").get_stats()`
*   **Lexical Search:** `result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal Search (Text + Image):** See detailed examples in the original README.
*   **Delete Documents:** `results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])`
*   **Delete Index:** `results = mq.index("my-first-index").delete()`

### Advanced Features

*   **Searching Using Weights in Queries:**  Create more advanced and complex semantic queries.
*   **Creating and Searching Indexes with Multimodal Combination Fields:** Combine text and images into one field.

### Running Marqo in Production

*   **Kubernetes:** Deploy Marqo on your cloud provider using our [Kubernetes templates](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed experience, sign up for [Marqo Cloud](https://cloud.marqo.ai).

### Documentation & Resources

*   **Full Documentation:** [https://docs.marqo.ai/](https://docs.marqo.ai/)
*   **Marqo for Image Data:** Building advanced image search with Marqo.
*   **Marqo for text:** Building a multilingual database in Marqo.
*   **Integrating Marqo with GPT:** Making GPT a subject matter expert by using Marqo as a knowledge base.
*   **Marqo for Creative AI:** Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.
*   **Marqo and Speech Data:** Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   **Marqo for content moderation:** Building advanced image search with Marqo to find and remove content.
*   **Getting started with Marqo Cloud:** Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo
*   **Marqo for e-commerce:** This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript.
*   **Marqo chatbot:** In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API.
*   **Core Features:** Marqo's core features.

### Support

*   **Discourse Forum:** Ask questions and share your creations on our [Discourse forum](https://community.marqo.ai).
*   **Slack Community:** Join our [Slack community](https://bit.ly/marqo-community-slack) to connect with other users and developers.

### Contribute

We welcome contributions! Please see [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.