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

**Marqo is a powerful, open-source vector search engine that simplifies building semantic search into your applications, handling everything from vector generation to retrieval.** Explore the [Marqo GitHub Repository](https://github.com/marqo-ai/marqo) to get started.

**Key Features:**

*   **🚀 State-of-the-Art Embeddings:** Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more, with both CPU and GPU support.
*   **⚡ High Performance:** Experience cutting-edge search speeds with in-memory HNSW indexes and scale to millions of documents. Benefit from asynchronous and non-blocking data upload and search.
*   **🌌 Documents-in, Documents-Out:** Simplify your workflow with built-in vector generation, storage, and retrieval for both text and images, enabling complex semantic queries and metadata storage.
*   **🍱 Managed Cloud Option:** Take advantage of a fully managed cloud service with low-latency deployments, scalability, high availability, and 24/7 support. Learn more at [Marqo Cloud](https://www.marqo.ai/cloud).
*   **🔌 Easy Integrations:** Seamlessly integrate with popular AI and data processing frameworks like Haystack, Griptape, Langchain, and Hamilton.

## Quick Start

Get up and running with Marqo in minutes.

1.  **Install Docker:**  Make sure you have Docker installed ([Docker Official Website](https://docs.docker.com/get-docker/)). Ensure Docker has at least 8GB memory and 50GB storage allocated.
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

4.  **Index and Search:**  Here's a basic example:

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
    *   `create_index()` creates an index (specify a model like `"hf/e5-base-v2"`).
    *   `add_documents()` adds documents to the index, specifying `tensor_fields` for vectorization.
    *   You can retrieve results using `search()` to find the best matches.
    *   Refer to [here](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) for the full list of models.

    **Example Results:**

    ```python
    import pprint
    pprint.pprint(results)
    ```

    (Output demonstrating the search results.)

    *   `hits` contains matching documents (ordered by relevance).
    *   `_highlights` shows the most relevant parts of each document.

## Core Operations

### Get Document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

### Get Index Stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi Modal and Cross Modal Search

Enable image and text search. Create an index with a CLIP configuration, as below:

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

### Searching using an image

Searching using an image can be achieved by providing the image link.

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching with Weighted Queries

Use weighted queries for more advanced and nuanced searches.

```python
# (See example code in original README)
```

### Creating and searching indexes with multimodal combination fields

```python
# (See example code in original README)
```

### Delete Documents

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

```python
results = mq.index("my-first-index").delete()
```

## Production Deployment

*   **Kubernetes:** Deploy Marqo using Kubernetes templates: [marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:**  Get a fully managed experience at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Learn More

| Resource | Description |
|---|---|
| [Quick Start](#quick-start) | Build your first application. |
| [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search. |
| [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database. |
| [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Augmenting GPT with Marqo. |
| [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Semantic search with Stable Diffusion. |
| [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Speech processing with Marqo. |
| [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Find and remove content with Marqo. |
| [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | How to get set up and running with Marqo Cloud. |
| [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Web application with frontend and backend. |
| [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Chatbot application using Marqo and OpenAI's ChatGPT API. |
| [Features](#-core-features) | Marqo's core features. |

## Documentation

Access the full documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important

Do not run other applications on Marqo's Vespa cluster; Marqo automatically modifies the cluster settings.

## Contribute

Marqo welcomes contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## Development Setup

```bash
python -m venv ./venv
source ./venv/bin/activate
pip install -r requirements.txt
tox
```

## Merge Instructions

1.  Run the full test suite (`tox`).
2.  Create a pull request with an associated GitHub issue.

## Support

*   **Discourse:** [Discourse Forum](https://community.marqo.ai)
*   **Slack:** [Slack Community](https://bit.ly/marqo-community-slack)