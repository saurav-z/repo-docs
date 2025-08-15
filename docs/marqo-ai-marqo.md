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

## Marqo: The Open-Source, End-to-End Vector Search Engine for Text and Images

Marqo simplifies building powerful vector search applications by offering vector generation, storage, and retrieval through a single API.  Explore the full capabilities on the [Marqo GitHub Repository](https://github.com/marqo-ai/marqo).

### **Key Features**

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize cutting-edge machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or easily integrate your own.
    *   Supports both CPU and GPU for optimized performance.

*   **⚡ High Performance & Scalability:**
    *   Leverage in-memory HNSW indexes for blazingly fast search speeds.
    *   Scale to indexes with hundreds of millions of documents using horizontal index sharding.
    *   Benefit from asynchronous and non-blocking data upload and search operations.

*   **🌌 Documents-in, Documents-out:**
    *   Vector generation, storage, and retrieval are provided out of the box.
    *   Build search, entity resolution, and data exploration applications with your text and images.
    *   Construct sophisticated semantic queries by combining weighted search terms.
    *   Filter search results with Marqo's query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

*   **🍱 Fully Managed Cloud (Optional):**
    *   Low latency optimized deployment of Marqo.
    *   Scale inference at the click of a button.
    *   High availability.
    *   24/7 support.
    *   Access control.
    *   Learn more [here](https://www.marqo.ai/cloud).

### **Integrations**

Marqo seamlessly integrates with popular AI and data processing frameworks, with more integrations constantly being developed.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

### **Get Started Quickly**

Follow these steps to begin using Marqo:

1.  **Install Docker:**  Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure at least 8GB of memory and 50GB of storage are allocated to Docker.
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

4.  **Index and Search:** See the code example in the original README, or explore the Getting Started guide on [Marqo's documentation](https://docs.marqo.ai/).

### **Learn More**

| Resource                                                                                                                                                                                                          | Description                                                                            |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)                                                                                                                                                                               | Build your first application with Marqo in under 5 minutes.                          |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)                                                                                        | Building advanced image search with Marqo.                                           |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)                                                                                     | Building a multilingual database in Marqo.                                            |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering)                            | Making GPT a subject matter expert by using Marqo as a knowledge base.                  |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs)                                                                 | Combining stable diffusion with semantic search to generate and categorise 100k images. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                                                                                                                                             | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.  |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)                                                                                        | Building advanced image search with Marqo to find and remove content.                |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)                                                                                                                  | Get up and running with Marqo Cloud.                                                   |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)                                                                                            | Web application demo with frontend and backend.                                        |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)                                                                                                               | Build a chat bot application using Marqo and OpenAI's ChatGPT API.                    |
| 🦾 [Features](#-Core-Features)                                                                                                                                                                                    | Marqo's core features.                                                                 |

### Running Marqo Open Source in Production

Deploy Marqo on your cloud provider of choice using Kubernetes templates ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)) for features like replicas, storage shards, and inference nodes. For a fully managed cloud service, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

### **Documentation**

For comprehensive information, visit the official Marqo documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

### **Important Note**

Avoid running other applications on Marqo's Vespa cluster, as Marqo dynamically adjusts the cluster settings.

### **Contributing**

Marqo is a community-driven project.  Learn how to contribute by reading [CONTRIBUTING.md](./CONTRIBUTING.md).

### **Development Setup**

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests: Execute `tox` within the project directory.
5.  If dependencies change, delete the `.tox` directory and rerun.

### **Merge Instructions**

1.  Run the complete test suite:  Use `tox` in the project directory.
2.  Create a pull request and link it to the associated GitHub issue.

### **Support**

*   Ask questions and share ideas on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack) for real-time discussions.