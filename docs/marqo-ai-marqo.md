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

## Marqo: The Open-Source Vector Search Engine for Text & Images

**Marqo simplifies building powerful search applications by providing an end-to-end vector search solution, from vector generation to retrieval, all through a single API.**  Find out more on the [Marqo GitHub](https://github.com/marqo-ai/marqo).

### Key Features:

*   **🤖 State-of-the-Art Embeddings:** Leverage cutting-edge machine learning models from PyTorch, Hugging Face, OpenAI, and more.  Supports both CPU and GPU.
*   **⚡ High Performance:** Benefit from in-memory HNSW indexes for blazing-fast search speeds.  Scales to hundreds of millions of documents with horizontal sharding.
*   **🌌 Documents-in, Documents-out:** Easily build search, entity resolution, and data exploration applications with text and images. Handle vector generation, storage, and retrieval out-of-the-box.
*   **🍱 Multimodal Search:**  Seamlessly search across text and images using CLIP models.
*   **☁️ Managed Cloud Option:** Easily scale inference with Marqo Cloud, which offers low-latency, high-availability, and 24/7 support.

### Core Functionality

Marqo is designed to handle the complexities of vector search, offering a seamless experience for developers:

*   **Integrated Embedding Generation:** Eliminate the need to manage separate embedding models and preprocessing steps. Marqo handles it all.
*   **Flexible Data Types:** Store both unstructured data and semi-structured metadata together in documents, including a range of data types such as bools, ints, and keywords.
*   **Advanced Querying:** Construct complex semantic queries by combining weighted search terms and filter results using Marqo's query DSL.

### Quick Start

Get started with Marqo in minutes:

1.  **Install Docker:** Follow instructions on the [Docker Official website](https://docs.docker.com/get-docker/), ensuring at least 8GB memory and 50GB storage are allocated.
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
    ```

    *   `mq` is the Marqo API client.
    *   `create_index()` creates a new index, and allows you to specify the model (e.g., `hf/e5-base-v2`).
    *   `add_documents()` takes a list of documents for indexing, with `tensor_fields` identifying fields to be indexed as vectors.
    *   Documents can have a custom ID with the `_id` field.

### Learn More

*   **[Quick start](#quick-start):** Build your first application with Marqo in under 5 minutes.
*   **[Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization):** Building advanced image search with Marqo.
*   **[Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code):** Building a multilingual database in Marqo.
*   **[Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering):** Making GPT a subject matter expert by using Marqo as a knowledge base.
*   **[Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs):** Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.
*   **[Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing):** Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   **[Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo):** Building advanced image search with Marqo to find and remove content.
*   **[Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud):** Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo
*   **[Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md):** This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.
*   **[Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo):** In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.
*   **[Features](#-core-features):** Marqo's core features.

### Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Use Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Leverage Marqo for scalable search within LLM-based agents.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Integrate Marqo with LangChain applications for vector search.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Use Marqo with Hamilton LLM applications.

### Additional Operations

*   **Get document:** Retrieve a document by ID.  `result = mq.index("my-first-index").get_document(document_id="article_591")`  Note that adding a document with an existing ID will update the document.
*   **Get index stats:** Get information about an index.  `results = mq.index("my-first-index").get_stats()`
*   **Lexical search:** Perform a keyword search. `result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)`
*   **Multimodal and cross modal search:**  Create an index with a CLIP configuration to search images.
    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```
    Images can then be added within documents as follows. You can use urls from the internet (for example S3) or from the disk of the machine:

    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```

    You can then search the image field using text.

    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```
    Searching using an image can be achieved by providing the image link.

    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```
*   **Searching using weights in queries:**  Use a dictionary to weight query terms.
    ```python
    query = {
        "I need to buy a communications device, what should I get?": 1.1,
        "The device should work like an intelligent computer.": 1.0,
    }
    ```
*   **Creating and searching indexes with multimodal combination fields:** Combine text and images into one field for more efficient searching.

### Production Deployment

*   **Kubernetes:** Deploy Marqo using our Kubernetes templates: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed experience, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

### Important Notes

*   Do not run other applications on Marqo's Vespa cluster.

### Contributing

We welcome contributions!  Please review our [CONTRIBUTING.md](./CONTRIBUTING.md) guidelines.

### Development Setup

1.  Create a virtual env: `python -m venv ./venv`
2.  Activate the env: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox`
5.  If updating dependencies, delete the `.tox` dir and rerun.

### Merge Instructions

1.  Run the full test suite (using `tox`).
2.  Create a pull request with an attached GitHub issue.

### Support

*   **Discourse:**  [https://community.marqo.ai](https://community.marqo.ai)
*   **Slack:** [https://bit.ly/marqo-community-slack](https://bit.ly/marqo-community-slack)