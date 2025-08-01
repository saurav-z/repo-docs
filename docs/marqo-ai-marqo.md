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

**Marqo is a powerful, open-source vector search engine that simplifies building search applications with text and images, offering vector generation, storage, and retrieval through a single API.**  Explore the original repository [here](https://github.com/marqo-ai/marqo).

**Key Features:**

*   🤖 **State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.  Choose from pre-configured models or bring your own, with support for CPU and GPU.
*   ⚡ **High Performance:** Benefit from in-memory HNSW indexes for cutting-edge search speeds. Scale to hundreds of millions of documents with horizontal index sharding and leverage async and non-blocking data upload and search.
*   🌌 **Documents-In, Documents-Out:** Simplify your workflow with out-of-the-box vector generation, storage, and retrieval.  Build search, entity resolution, and data exploration applications using both text and images. Construct complex semantic queries by combining weighted search terms and filter results using Marqo’s query DSL.
*   🍱 **Managed Cloud (Optional):** Access a low-latency optimized deployment of Marqo with scaling at the click of a button.  Enjoy high availability, 24/7 support, access control, and more via [Marqo Cloud](https://www.marqo.ai/cloud).
*   🖼 **Multimodal Search:** Easily index and search images alongside text with CLIP models.
*   ⚖️ **Weighted Queries:** Refine search results by applying weights to queries, for more targeted searches.

## Getting Started

Follow these steps to quickly get started with Marqo:

1.  **Docker Setup:** Ensure Docker is installed. Refer to the [Docker Official website](https://docs.docker.com/get-docker/) for installation instructions. Docker should have at least 8GB of memory and 50GB of storage allocated.
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
5.  **Examine Results:**
    ```python
    import pprint
    pprint.pprint(results)
    ```

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   💙 [Haystack](https://github.com/deepset-ai/haystack)
*   🛹 [Griptape](https://github.com/griptape-ai/griptape)
*   🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)
*   ⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)

## Learn More

Explore these resources for deeper insights:

| Resource                                                                                                                                                                                             | Description                                                                                                        |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------ |
| 📗 [Quick start](#getting-started)                                                                                                                                                                 | Build your first application with Marqo in under 5 minutes.                                                        |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)                                                                          | Building advanced image search with Marqo.                                                                        |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)                                                                        | Building a multilingual database in Marqo.                                                                        |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering)               | Making GPT a subject matter expert by using Marqo as a knowledge base.                                            |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs)                                                    | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.                  |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                                                                                                                            | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                             |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)                                                                      | Building advanced image search with Marqo to find and remove content.                                            |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)                                                                                                      | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)                                                                           | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript.         |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)                                                                                               | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API.                            |
| 🦾 [Features](#-core-features)                                                                                                                                                                      | Marqo's core features.                                                                                             |

## Production Deployment

Marqo supports Kubernetes templates for deployment on your preferred cloud provider.  Find the repository [here](https://github.com/marqo-ai/marqo-on-kubernetes).  For a fully managed cloud service, sign up for [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Contributing

Marqo is a community-driven project.  Review the [CONTRIBUTING.md](./CONTRIBUTING.md) file to learn how to contribute.

## Support

*   Engage with the community on our [Discourse forum](https://community.marqo.ai).
*   Join the [Slack community](https://bit.ly/marqo-community-slack) for discussions and support.