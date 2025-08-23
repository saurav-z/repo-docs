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

# Marqo: The End-to-End Vector Search Engine for Text and Images

**Marqo simplifies vector search by providing vector generation, storage, and retrieval in a single, easy-to-use API.**  [Explore the original repo](https://github.com/marqo-ai/marqo).

## Key Features

*   **🚀 State-of-the-Art Embeddings:**
    *   Leverage the latest machine learning models from Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU acceleration.

*   **⚡ High Performance:**
    *   Utilizes in-memory HNSW indexes for blazingly fast search speeds.
    *   Scales to handle indexes with hundreds of millions of documents through horizontal index sharding.
    *   Offers async and non-blocking data upload and search capabilities.

*   **📦 Documents-In, Documents-Out:**
    *   Handles vector generation, storage, and retrieval out-of-the-box.
    *   Build powerful search, entity resolution, and data exploration applications using your text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo's flexible query DSL.
    *   Store both unstructured data and semi-structured metadata seamlessly.

*   **☁️ Managed Cloud:**
    *   Low-latency optimized deployment of Marqo.
    *   Scale your inference at the click of a button.
    *   Ensured high availability with 24/7 support.
    *   Access control included.
    *   Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack):** Integrate Marqo as your Document Store for NLP pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape):** Provide your LLM-based agents access to scalable search.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain):** Leverage Marqo for your LangChain applications with a vector search component.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Use Marqo for Hamilton LLM applications.

## Get Started Quickly

Follow these steps to get up and running with Marqo in minutes:

1.  **Install Docker:**  Ensure you have Docker installed.  Refer to the [Docker Official website](https://docs.docker.com/get-docker/) for installation instructions.  Allocate at least 8GB of memory and 50GB of storage to Docker.

2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```

3.  **Install the Marqo Client:**

    ```bash
    pip install marqo
    ```

4.  **Start Indexing and Searching:**  Here's a basic example:

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

## Learn More

Explore these resources to dive deeper into Marqo:

*   📗 [Quick start](#getting-started): Build your first application with Marqo.
*   🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization): Advanced image search.
*   📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code): Building a multilingual database.
*   🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering): Augmenting GPT.
*   🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs): Combining stable diffusion and semantic search.
*   🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing): Add diarization and transcription.
*   🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo): Content moderation with Marqo.
*   ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud): Set up and run with Marqo Cloud.
*   👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md): E-commerce demo.
*   🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo): Build a chatbot.
*   🦾 [Core Features](#-core-features): Learn about Marqo's core features.

## Running Marqo in Production

*   **Kubernetes:** Utilize our Kubernetes templates for deployment on your preferred cloud provider. Explore [marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** Experience a fully managed cloud service at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Access comprehensive documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Contributing

Marqo is a community-driven project. We welcome contributions!  Read our [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## Support

*   Join the community on our [Discourse forum](https://community.marqo.ai).
*   Connect with us on [Slack](https://bit.ly/marqo-community-slack).