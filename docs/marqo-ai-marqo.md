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

**Marqo simplifies building powerful search applications by providing an end-to-end vector search engine for both text and images with a single API.**

### Key Features:

*   **State-of-the-Art Embeddings:** Leverages cutting-edge machine learning models from Hugging Face, OpenAI, and more, with both CPU and GPU support.
*   **High Performance:** Achieves blazing-fast search speeds with embeddings stored in in-memory HNSW indexes and scales to millions of documents.
*   **Documents-in, Documents-out:** Simplifies the process with built-in vector generation, storage, and retrieval for text and images.
*   **Multimodal Search:** Seamlessly search across both text and images, including image URLs and local files.
*   **Weighted Queries:** Offers flexible search capabilities, allowing you to create complex semantic queries with weighted terms and filtering options.
*   **Open Source and Cloud Options:** Choose from a fully open-source, self-hosted solution or a managed cloud service.
*   **Easy Integrations:** Works seamlessly with popular AI frameworks, including Haystack, Griptape, Langchain, and Hamilton.
*   **Multimodal Combination Fields:** Combine text and images into a single field for more efficient storage and search.

### Get Started Quickly

1.  **Prerequisites:** Requires Docker. Ensure Docker has at least 8GB memory and 50GB storage.
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
4.  **Index and Search:**
    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')
    mq.create_index("my-first-index", model="hf/e5-base-v2")
    mq.index("my-first-index").add_documents([
        {"Title": "The Travels of Marco Polo", "Description": "A 13th-century travelogue describing Polo's travels"}, 
        {"Title": "Extravehicular Mobility Unit (EMU)", "Description": "The EMU is a spacesuit..." , "_id": "article_591"}
        ], tensor_fields=["Description"])
    results = mq.index("my-first-index").search(q="What is the best outfit to wear on the moon?")
    ```
    For more detailed instructions, please refer to the [Getting Started](#getting-started) section of this README, or view the official [Marqo documentation](https://docs.marqo.ai/).

### Additional Operations:

*   **Get Document:** `mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** `mq.index("my-first-index").get_stats()`
*   **Lexical Search:** `mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal Search (Image Search):**  Create an index with `settings = {"treat_urls_and_pointers_as_images":True, "model":"ViT-L/14"}`.  Add images as URLs within documents. Search using text or image URLs.
*   **Weighted Queries:** Utilize dictionaries to weight search terms for more nuanced queries.
*   **Delete Documents:** `mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])`
*   **Delete Index:** `mq.index("my-first-index").delete()`

### Production Deployment

Marqo supports Kubernetes deployments for production environments. Find Kubernetes templates at: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).

For a fully managed cloud service, visit [https://cloud.marqo.ai](https://cloud.marqo.ai).

### Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   [Haystack](https://github.com/deepset-ai/haystack)
*   [Griptape](https://github.com/griptape-ai/griptape)
*   [Langchain](https://github.com/langchain-ai/langchain)
*   [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)

### Learn More

*   **Quick Start:** Build your first application in minutes.
*   **Marqo for Image Data:** Building advanced image search.
*   **Marqo for Text:** Building a multilingual database.
*   **Integrating Marqo with GPT:** Augmenting GPT with Marqo for context-aware question answering.
*   **Marqo for Creative AI:** Combining Stable Diffusion with semantic search.
*   **Marqo and Speech Data:** Add diarisation and transcription for Q&A.
*   **Marqo for Content Moderation:** Find and remove undesirable content.
*   **Getting Started with Marqo Cloud:** Getting started with Marqo Cloud.
*   **Marqo for e-commerce:** A web application with a frontend and backend using Python, Flask, ReactJS, and Typescript.
*   **Marqo chatbot:** In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API.
*   **Core Features:** Marqo's core features.

### Additional Resources

*   **Documentation:** Comprehensive guides and API reference: [https://docs.marqo.ai/](https://docs.marqo.ai/).
*   **Kubernetes Deployment:** Deploy Marqo in production: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** Fully managed vector search: [https://cloud.marqo.ai](https://cloud.marqo.ai)
*   **Marqo Blog:** Stay up-to-date with the latest features and use cases: [https://www.marqo.ai/blog](https://www.marqo.ai/blog)

### Contribute

Marqo is a community project.  Learn how to contribute [here](./CONTRIBUTING.md)

### Support

*   **Discourse Forum:** Ask questions and share your ideas: [https://community.marqo.ai](https://community.marqo.ai)
*   **Slack Community:** Connect with other users and the team: [https://bit.ly/marqo-community-slack](https://bit.ly/marqo-community-slack)

### Development Setup

Follow these steps to set up your development environment.

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the environment: `source ./venv/bin/activate`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run tests with `tox`.
5.  Update dependencies and rerun tests after deleting the `.tox` directory.

### Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request linked to a GitHub issue.

Find the original repository on GitHub: [https://github.com/marqo-ai/marqo](https://github.com/marqo-ai/marqo)