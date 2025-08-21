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

## Marqo: The Open-Source Vector Search Engine for Text and Images

**Marqo** is a powerful, open-source vector search engine that simplifies semantic search for both text and images, making it easy to build intelligent applications.  Explore the full capabilities on the [Marqo GitHub Repository](https://github.com/marqo-ai/marqo).

### **Key Features**

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more. Support for both CPU and GPU.
*   **⚡ High Performance:**  Benefit from in-memory HNSW indexes for blazingly fast search speeds, scaling to hundreds of millions of documents with horizontal sharding and asynchronous data handling.
*   **🌌 Documents-In-Documents-Out:** Simplify your workflow with built-in vector generation, storage, and retrieval for text and images. Easily build applications for search, entity resolution, and data exploration.
*   **🍱 Managed Cloud Option:** Leverage a fully managed cloud service with low-latency optimization, scalable inference, high availability, 24/7 support, and access control. (Learn more: [Marqo Cloud](https://www.marqo.ai/cloud))
*   **🖼️ Multimodal Search**: Easily search images by text, and text by images.  
*   **🎨 Multimodal Combination Fields**: Combine text and images into a single field for more advanced search capabilities. 
*   **⚖️ Weighted Queries:** Construct advanced semantic queries by combining weighted search terms and utilizing Marqo’s query DSL.

### **Why Choose Marqo?**

Marqo goes beyond a basic vector database by offering a complete end-to-end vector search solution. It handles everything from vector generation and model deployment to preprocessing and metadata management, enabling developers to quickly integrate powerful search capabilities without the complexities of managing multiple components.

### **Getting Started: Quick Installation & Example**

Follow these steps to get up and running with Marqo:

1.  **Install Docker:** ( [Docker Official website](https://docs.docker.com/get-docker/) ) Ensure Docker has at least 8GB memory (recommended).
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
4.  **Start Indexing and Searching!**

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

### **Integrations**

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

### **Learn More**

|                                                        |                                                                                                                                                                                                                                                                                                 |
| ------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#getting-started)                   | Build your first application with Marqo in under 5 minutes.                                                                                                                                                                                                                                   |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo.                                                                                                                                                                                                                                               |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo.                                                                                                                                                                                                                                             |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.                                                                                                                                                                                                |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.                                                                                                                                                                                |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                                                                                                                                             |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content.                                                                                                                                                                                                             |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API. |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour. |
| 🦾 [Features](#-core-features)                        | Marqo's core features.                                                                                                                                                                                                                                                                          |

### **Advanced Operations**

*   **Get Document:** Retrieve a document by its ID.
*   **Get Index Stats:** View information about an index.
*   **Lexical Search:** Perform keyword searches.
*   **Multi-modal & Cross-modal Search:**  Search images by text, and text by images, using CLIP models.
*   **Searching using Weights:**  Create sophisticated queries with weighted components.
*   **Multimodal Combination Fields:** Search multimodal fields combining text and images.
*   **Delete Documents:** Remove documents by ID.
*   **Delete Index:** Delete an entire index.

### **Running Marqo in Production**

*   **Kubernetes:** Deploy Marqo using Kubernetes templates.  ([Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes))
*   **Marqo Cloud:** Fully managed cloud service ([https://cloud.marqo.ai](https://cloud.marqo.ai)).

### **Important Notes**

*   Do not run other applications on Marqo's Vespa cluster.

### **Contributing**

Marqo is a community-driven project! We welcome contributions.  See our [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

### **Development Setup**

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the virtual environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox` (within the project directory)
5.  If updating dependencies, delete the `.tox` directory and rerun `tox`.

### **Merge Instructions**

1.  Run the full test suite (`tox`).
2.  Create a pull request with an attached GitHub issue.

### **Support**

*   **Discourse:** Ask questions and share on our [Discourse forum](https://community.marqo.ai).
*   **Slack:** Join our [Slack community](https://bit.ly/marqo-community-slack).