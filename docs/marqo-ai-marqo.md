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

# Marqo: The Open-Source Vector Search Engine for Text & Images

**Marqo simplifies building powerful search applications by providing end-to-end vector search capabilities in a single API.**  [Explore the Marqo repository on GitHub](https://github.com/marqo-ai/marqo).

## Key Features

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize cutting-edge machine learning models from Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU acceleration.

*   **⚡ High Performance:**
    *   Leverage in-memory HNSW indexes for lightning-fast search speeds.
    *   Scale to indexes containing hundreds of millions of documents with horizontal index sharding.
    *   Benefit from asynchronous and non-blocking data upload and search operations.

*   **🌌 Documents-In-Documents-Out:**
    *   Get vector generation, storage, and retrieval all in one place.
    *   Build search, entity resolution, and data exploration applications for both text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo’s query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

*   **🍱 Managed Cloud (Optional):**
    *   Low-latency, optimized deployment of Marqo.
    *   Scale inference with ease.
    *   High availability.
    *   24/7 support.
    *   Access control.
    *   Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).

## Integrations

Marqo integrates with popular AI and data processing frameworks, with more on the way.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More About Marqo

|                                                                                      |                                                                                                                                                                                                                            |
| --------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick Start](#getting-started)                                                     | Build your first application with Marqo in under 5 minutes.                                                                                                                                                                 |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo.                                                                                                                                                                    |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo.                                                                                                                                                                      |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.                                                                                                                                                      |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.                                                                                                                       |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                                                                                       |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content.                                                                                                                                     |
| ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Learn how to get set up and running with Marqo Cloud, from initial login to building your first application with Marqo.                                                                                                     |
| 👗 [Marqo for E-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API. |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chat bot application using Marqo and OpenAI's ChatGPT API. Learn how to customize the behavior.                                                                                                           |
| 🦾 [Core Features](#-core-features) | Marqo's core features. |

## Getting Started

1.  **Prerequisites**: Marqo requires Docker.
    *   Install Docker: Visit the [Docker Official Website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB memory and 50GB storage.

2.  **Run Marqo using Docker:**

```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
```

    *   *Note:*  If the `marqo` container is killed frequently, increase the memory limit for Docker to at least 6GB (8GB recommended) in your Docker settings.

3.  **Install the Marqo Client:**

```bash
pip install marqo
```

4.  **Start Indexing and Searching:** Here's a basic example:

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.create_index("my-first-index", model="hf/e5-base-v2")  # Or your preferred model

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

*   `mq`: The client that interacts with the Marqo API.
*   `create_index()`: Creates a new index.  You can specify the embedding model (e.g., "hf/e5-base-v2").  Experimentation is often needed to find the best model for your use case.
*   `add_documents()`:  Indexes documents (represented as Python dicts). `tensor_fields` specify fields for vector indexing.
*   You can set a document's ID with the `_id` field; otherwise, Marqo generates one.

5.  **View Results:**

```python
import pprint
pprint.pprint(results)
```

Results will include hits with `_highlights` showing the matching parts of documents,  ordered by relevance (`_score`).  `limit` controls the number of results.

## Other Basic Operations

### Get Document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*   *Note:*  Adding a document with the same `_id` updates the existing document.

### Get Index Stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi-Modal and Cross-Modal Search

*   **Image Search**: Marqo supports image and text search using CLIP models.

    1.  Create an index configured for images:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

    2.  Add images (using URLs or local file paths) within documents:

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

    3.  Search using text:

```python
results = mq.index("my-multimodal-index").search('animal')
```

    4.  Search using an image URL:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching Using Weights in Queries

```python
# weighted queries example
```

### Creating and Searching Indexes with Multimodal Combination Fields

```python
# multimodal combination fields example
```

### Delete Documents

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo Open Source in Production

*   **Kubernetes**:  Deploy Marqo using Kubernetes templates ([https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)).
*   **Marqo Cloud**: For a fully managed service, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

*   Find the full Marqo documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Note

*   Do *not* run other applications on Marqo's Vespa cluster, as Marqo manages the cluster settings automatically.

## Contributing

Marqo thrives on community contributions! See  [CONTRIBUTING.md](./CONTRIBUTING.md) for details on how to get involved.

## Dev Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate the environment:  `source ./venv/bin/activate`
3.  Install dependencies:  `pip install -r requirements.txt`
4.  Run tests:  `tox` (from the project root)
5.  If dependencies change, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.

## Support

*   **Discourse:** Ask questions and share your work: [https://community.marqo.ai](https://community.marqo.ai).
*   **Slack:** Join the community: [https://bit.ly/marqo-community-slack](https://bit.ly/marqo-community-slack).