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

## Marqo: The Open-Source, End-to-End Vector Search Engine for Text and Images

Marqo simplifies vector search by handling vector generation, storage, and retrieval with a single API, making it easy to build powerful search applications.  [See the original repo](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more, with both CPU and GPU support.
*   **High Performance:** Experience cutting-edge search speeds with embeddings stored in in-memory HNSW indexes, and scale to handle hundreds of millions of documents with horizontal index sharding.
*   **Documents-In, Documents-Out:** Build applications using text and images with out-of-the-box vector generation, storage, and retrieval, easily building search, entity resolution, and data exploration applications.
*   **Multimodal Search:** Seamlessly search both text and images, including support for cross-modal search.
*   **Flexibility and Control:** Choose from pre-configured models or bring your own, and customize search behavior.
*   **Managed Cloud Option:**  Deploy and scale with ease with Marqo Cloud, offering low-latency deployment, scalability, high availability, and 24/7 support.

## Why Choose Marqo?

Marqo goes beyond a standard vector database by integrating machine learning model management and input preprocessing, enabling developers to build sophisticated search capabilities with minimal effort. It simplifies the complexities of vector search, offering a streamlined "documents in, documents out" approach.

## Getting Started

1.  **Docker:** Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/).  Ensure Docker has at least 8GB memory and 50GB storage.
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
4.  **Example Code:**

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

## Core Features (Detailed)

*   **🤖 State of the art embeddings**
    *   Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more. 
    *   Start with a pre-configured model or bring your own.
    *   CPU and GPU support.
*   **⚡ Performance**
    *   Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
    *   Scale to hundred-million document indexes with horizontal index sharding.
    *   Async and non-blocking data upload and search.
*   **🌌 Documents-in-documents-out**
    *   Vector generation, storage, and retrieval are provided out of the box.
    *   Build search, entity resolution, and data exploration application with using your text and images.
    *   Build complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo’s query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.
*   **🍱 Managed cloud**
    *   Low latency optimised deployment of Marqo.
    *   Scale inference at the click of a button.
    *   High availability.
    *   24/7 support.
    *   Access control.
    *   Learn more [here](https://www.marqo.ai/cloud).

## Integrations

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More

| | |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)| Build your first application with Marqo in under 5 minutes. |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo. |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base. |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT. |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content. |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo|
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.|
| 🦾 [Features](#-Core-Features) | Marqo's core features. |

## Other Basic Operations

*   **Get Document:**  `result = mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** `results = mq.index("my-first-index").get_stats()`
*   **Lexical Search:** `result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal and Cross-Modal Search:** Create an index with image settings (e.g., `settings = {"treat_urls_and_pointers_as_images":True, "model":"ViT-L/14"}`) and add image URLs to documents.
*   **Search with Images:** Search by image URL: `results = mq.index("my-multimodal-index").search('https://.../image.png')`
*   **Weighted Queries:**  Use dictionaries to specify query weights for more complex searches.
*   **Multimodal Combination Fields:**  Combine text and images within the same index field, specifying weights for each modality.
*   **Delete Documents:**  `results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])`
*   **Delete Index:** `results = mq.index("my-first-index").delete()`

## Running Marqo in Production

Marqo offers Kubernetes templates for deployment on your chosen cloud provider, supporting clusters with replicas, storage shards, and inference nodes ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)).  For a fully managed solution, explore Marqo Cloud: [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Notes

Avoid running other applications on Marqo's Vespa cluster, as Marqo automatically adjusts the cluster's settings.

## Contributing

Marqo is a community project. We welcome your contributions!  See [this](./CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the environment: `source ./venv/bin/activate`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run tests: `tox`.
5.  If dependencies change, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite (`tox`).
2.  Create a pull request with an attached GitHub issue.

## Support

*   [Discourse forum](https://community.marqo.ai): Ask questions and share your work.
*   [Slack community](https://bit.ly/marqo-community-slack): Chat with other community members.