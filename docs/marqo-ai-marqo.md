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

Marqo is an open-source, end-to-end vector search engine that simplifies building search, entity resolution, and data exploration applications for text and images.  [Explore Marqo on GitHub](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **🚀 Cutting-Edge Embeddings:** Leverages the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more; supports both CPU and GPU.
*   **⚡ Blazing Fast Performance:** Utilizes in-memory HNSW indexes for rapid search speeds and scales to millions of documents with horizontal index sharding.
*   **📦 Documents-in, Documents-out:** Simplifies vector generation, storage, and retrieval for text and images, allowing you to build complex semantic queries with ease.
*   **✨ Multimodal Search:**  Seamlessly integrate and search across text and images using CLIP models.
*   **☁️ Managed Cloud Option:** Provides a low-latency, optimized deployment of Marqo, with scalable inference, high availability, and 24/7 support.

**Why Choose Marqo?**

Marqo goes beyond a basic vector database by providing a complete solution for vector search.  It handles machine learning model management, preprocessing, and input transformations, enabling developers to quickly integrate powerful search capabilities with minimal effort.  Marqo supports many data types, and also lets you combine unstructured data and metadata in the same documents for complex search queries.

**Quick Start**

Get started with Marqo in just a few steps:

1.  **Install Docker:**  Follow the [Docker Official website](https://docs.docker.com/get-docker/) instructions to install Docker. Ensure Docker has at least 8GB memory and 50GB storage.
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
4.  **Start Indexing and Searching:**  See the example code in the original README.

**Integrations**

Marqo integrates with popular AI and data processing frameworks, including:

*   [Haystack](https://github.com/deepset-ai/haystack)
*   [Griptape](https://github.com/griptape-ai/griptape)
*   [Langchain](https://github.com/langchain-ai/langchain)
*   [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)

**Learn More:**

*   [Quick Start](#getting-started) - Build your first application with Marqo in under 5 minutes.
*   [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) - Building advanced image search with Marqo.
*   [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) - Building a multilingual database in Marqo.
*   [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) - Making GPT a subject matter expert by using Marqo as a knowledge base.
*   [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) - Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.
*   [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) - Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) - Building advanced image search with Marqo to find and remove content.
*   [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) - Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo
*   [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) - This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.
*   [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) - In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.
*   [Core Features](#-Core-Features) - Marqo's core features.

**Getting Started**

[See Quick Start instructions above](#getting-started)

**Other Basic Operations**
*   Get document
    *   Retrieve a document by ID.
*   Get index stats
    *   Get information about an index.
*   Lexical search
    *   Perform a keyword search.
*   Multi modal and cross modal search
    *   To power image and text search, Marqo allows users to plug and play with CLIP models from HuggingFace. **Note that if you do not configure multi modal search, image urls will be treated as strings.**
*   Searching using an image
    *   Searching using an image can be achieved by providing the image link.
*   Searching using weights in queries
    *   Queries can also be provided as dictionaries where each key is a query and their corresponding values are weights.
*   Creating and searching indexes with multimodal combination fields
    *   Marqo lets you have indexes with multimodal combination fields.
*   Delete documents
    *   Delete documents.
*   Delete index
    *   Delete an index.


**Running Marqo in Production**

Marqo provides Kubernetes templates for deployment on your preferred cloud provider.  For a fully managed cloud service, explore [Marqo Cloud](https://cloud.marqo.ai).

**Documentation**

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

**Contributing**

Marqo is a community-driven project.  Learn how to contribute by reading [this](./CONTRIBUTING.md).

**Support**

*   [Discourse Forum](https://community.marqo.ai) - Ask questions and share your creations.
*   [Slack Community](https://bit.ly/marqo-community-slack) - Chat with other community members.