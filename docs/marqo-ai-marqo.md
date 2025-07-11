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

**Unlock the power of semantic search with Marqo, an end-to-end vector search engine that simplifies indexing, storage, and retrieval for both text and image data.**  ([See the original repo](https://github.com/marqo-ai/marqo))

Marqo streamlines the complex process of building vector search into your applications, offering a seamless experience with its integrated features. 

**Key Features:**

*   **State-of-the-Art Embeddings:**
    *   Utilize cutting-edge machine learning models from PyTorch, Huggingface, OpenAI, and more.
    *   Easily integrate pre-configured or custom models.
    *   Benefit from both CPU and GPU support.

*   **High-Performance Search:**
    *   Achieve rapid search speeds with in-memory HNSW indexes.
    *   Scale effortlessly to handle indexes with hundreds of millions of documents through horizontal index sharding.
    *   Experience efficient asynchronous and non-blocking data upload and search.

*   **Documents-In, Documents-Out:**
    *   Simplify your workflow with out-of-the-box vector generation, storage, and retrieval.
    *   Create powerful search, entity resolution, and data exploration applications using your text and images.
    *   Build sophisticated semantic queries with weighted search terms.
    *   Filter search results using Marqo's query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

*   **Managed Cloud Solution:**
    *   Benefit from a low-latency, optimized deployment of Marqo.
    *   Scale inference with a simple click.
    *   Ensure high availability and 24/7 support.
    *   Take advantage of access control features.
    *   Learn more about Marqo Cloud [here](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **[Haystack](https://github.com/deepset-ai/haystack):** Use Marqo as your Document Store for NLP pipelines.
*   **[Griptape](https://github.com/griptape-ai/griptape):** Give LLM-based agents access to scalable search with your data.
*   **[Langchain](https://github.com/langchain-ai/langchain):** Leverage open source or custom fine tuned models through Marqo for LangChain applications.
*   **[Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Leverage open source or custom fine tuned models through Marqo for Hamilton LLM applications.

## Getting Started

1.  **Install Docker:** Follow the instructions on the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage.

2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

    *Note: If your `marqo` container gets killed, increase Docker memory to at least 6GB (8GB recommended).*

3.  **Install the Marqo client:**

    ```bash
    pip install marqo
    ```

4.  **Start indexing and searching:**

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

    import pprint
    pprint.pprint(results)
    ```

## Additional Operations

*   **Get Document:** `result = mq.index("my-first-index").get_document(document_id="article_591")`
*   **Get Index Stats:** `results = mq.index("my-first-index").get_stats()`
*   **Lexical Search:** `result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multi Modal Search:** (requires CLIP configuration)
*   **Delete Documents:** `results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])`
*   **Delete Index:** `results = mq.index("my-first-index").delete()`

## Production Deployment

*   **Kubernetes:**  Use Marqo's Kubernetes templates: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:**  Sign up for a fully managed cloud service: [https://cloud.marqo.ai](https://cloud.marqo.ai)

## Important Considerations

*   Do *not* run other applications on Marqo's Vespa cluster, as Marqo manages the cluster settings.

## Learn More

| Resource                                                                                                               | Description                                                                     |
| ---------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------- |
| 📗 [Quick start](#getting-started)                                                                                      | Build your first application with Marqo in under 5 minutes.                    |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo.                                       |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo.                                      |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.      |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.  |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.  |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content.          |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud.                |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) |  Web application with frontend and backend using Python, Flask, ReactJS, and Typescript  |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chat bot application using Marqo and OpenAI's ChatGPT API.          |
| 🦾 [Features](#-core-features)                                                                                             | Marqo's core features.                                                          |

##  Resources

*   **Documentation:** [https://docs.marqo.ai/](https://docs.marqo.ai/)
*   **Discourse Forum:**  [https://community.marqo.ai](https://community.marqo.ai)
*   **Slack Community:** [https://bit.ly/marqo-community-slack](https://bit.ly/marqo-community-slack)

## Contribute

Marqo is a community-driven project.  We welcome contributions!  See the [CONTRIBUTING.md](./CONTRIBUTING.md) file.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the environment: `source ./venv/bin/activate`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run tests: `tox` (from the root directory).
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite: `tox`.
2.  Create a pull request with a linked GitHub issue.