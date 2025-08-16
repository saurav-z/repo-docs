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

# Marqo: The Open-Source Vector Search Engine for Text and Images

**Looking for a fast, easy-to-use vector search solution that handles everything from embedding generation to retrieval?** Marqo is a powerful, open-source vector search engine designed for both text and images, making it simple to build intelligent search applications.  [Back to original repo](https://github.com/marqo-ai/marqo)

## Key Features

*   **🤖 State-of-the-Art Embeddings:**
    *   Leverage the latest machine learning models from Hugging Face, OpenAI, and more.
    *   Start with pre-configured models or bring your own custom models.
    *   Supports both CPU and GPU for optimal performance.
*   **⚡ High Performance:**
    *   Utilizes in-memory HNSW indexes for blazing-fast search speeds.
    *   Scales to hundreds of millions of documents with horizontal index sharding.
    *   Offers async and non-blocking data upload and search for improved efficiency.
*   **🌌 Documents-In-Documents-Out:**
    *   Handles vector generation, storage, and retrieval out-of-the-box.
    *   Easily build search, entity resolution, and data exploration applications using your text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo’s flexible query DSL.
    *   Store unstructured data and semi-structured metadata together in documents.
*   **🍱 Managed Cloud (Optional):**
    *   Low-latency optimized deployment of Marqo.
    *   Scale inference with ease.
    *   High availability.
    *   24/7 support.
    *   Access control.
    *   Learn more at [Marqo Cloud](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack):** Use Marqo as your Document Store for building NLP applications, including retrieval-augmentation, question answering, and document search.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape):** Empower LLM-based agents with scalable search capabilities using your own data.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain):** Integrate Marqo into your LangChain applications for vector search, using open-source or custom models.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Leverage Marqo for Hamilton LLM applications.

## Learn More About Marqo

Explore these resources to delve deeper into Marqo's capabilities:

| Resource                                                                                           | Description                                                                                                                                             |
| :------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#getting-started)                                                              | Build your first application with Marqo in minutes.                                                                                                     |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Learn how to perform advanced image searches with Marqo.                                                                                           |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Explore building a multilingual database using Marqo.                                                                                              |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Enhance GPT's knowledge base with Marqo for context-aware question answering.                                                                             |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine Stable Diffusion with semantic search to generate and categorize images.                                                                         |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                             | Add diarization and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                    |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Find and remove content with Marqo to build an image moderation system.                                                                             |
| ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)     | Learn to set up and run Marqo Cloud from initial login to building your first application.                                                               |
| 👗 [Marqo for E-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Build an e-commerce web application using Python, Flask, ReactJS, and Typescript.                                                                  |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)      | Build a chat bot application using Marqo and OpenAI's ChatGPT API.                                                                                    |
| 🦾 [Core Features](#-core-features)                                                                  | Explore Marqo's core features.                                                                                                                            |

## Getting Started

Follow these steps to quickly get up and running with Marqo:

1.  **Install Docker:** Go to the [Docker Official website](https://docs.docker.com/get-docker/) to install Docker. Ensure Docker has at least 8GB of memory (recommended) and 50GB of storage allocated.

2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your `marqo` container keeps getting killed, increase the memory limit for Docker to at least 6GB (8GB recommended) in your Docker settings.*

3.  **Install the Marqo Client:**

    ```bash
    pip install marqo
    ```

4.  **Start Indexing and Searching:** Here's a basic example:

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2")  # or another model

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

    # Print the results
    import pprint
    pprint.pprint(results)
    ```

    *   `mq`: This is the client that wraps the `marqo` API.
    *   `create_index()`: Creates a new index.  You can specify the model (e.g., `hf/e5-base-v2`).
    *   `add_documents()`: Adds documents (represented as Python dictionaries) for indexing. `tensor_fields` specifies which fields to index as vector collections.
    *   You can optionally set a document's ID with the special `_id` field; otherwise, Marqo generates one.

    Example results output:

    ```
    {
        'hits': [
            {
                'Title': 'Extravehicular Mobility Unit (EMU)',
                'Description': 'The EMU is a spacesuit that provides environmental protection, mobility, life support, and'
                               'communications for astronauts',
                '_highlights': [{
                    'Description': 'The EMU is a spacesuit that provides environmental protection, '
                                   'mobility, life support, and communications for astronauts'
                }],
                '_id': 'article_591',
                '_score': 0.61938936
            },
            {
                'Title': 'The Travels of Marco Polo',
                'Description': "A 13th-century travelogue describing Polo's travels",
                '_highlights': [{'Title': 'The Travels of Marco Polo'}],
                '_id': 'e00d1a8d-894c-41a1-8e3b-d8b2a8fce12a',
                '_score': 0.60237324
            }
        ],
        'limit': 10,
        'processingTimeMs': 49,
        'query': 'What is the best outfit to wear on the moon?'
    }
    ```

    *   Each `hit` is a matching document.
    *   Documents are ordered by relevance (`_score`).
    *   `_highlights` show the matching parts of documents.

## Other Basic Operations

### Get Document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*Note: Adding a document using `add_documents` with the same `_id` will update the document.*

### Get Index Stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi-Modal and Cross-Modal Search

Marqo lets you easily perform image and text search with CLIP models from Hugging Face. **Important:**  If multimodal search isn't configured, image URLs are treated as strings. Create an index with a CLIP configuration like so:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Then add images to your documents:

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

Search the image field using text:

```python
results = mq.index("my-multimodal-index").search('animal')
```

### Searching Using an Image

Search with an image link directly:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching with Weights in Queries

Use weighted queries for advanced searches:

```python
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

mq.create_index("my-weighted-query-index")

mq.index("my-weighted-query-index").add_documents(
    [
        {
            "Title": "Smartphone",
            "Description": "A smartphone is a portable computer device that combines mobile telephone "
            "functions and computing functions into one unit.",
        },
        {
            "Title": "Telephone",
            "Description": "A telephone is a telecommunications device that permits two or more users to"
            "conduct a conversation when they are too far apart to be easily heard directly.",
        },
        {
            "Title": "Thylacine",
            "Description": "The thylacine, also commonly known as the Tasmanian tiger or Tasmanian wolf, "
            "is an extinct carnivorous marsupial."
            "The last known of its species died in 1936.",
        }
    ],
    tensor_fields=["Description"]
)

# Example 1: Communications device
query = {
    "I need to buy a communications device, what should I get?": 1.1,
    "The device should work like an intelligent computer.": 1.0,
}
results = mq.index("my-weighted-query-index").search(q=query)

print("Query 1:")
pprint.pprint(results)

# Example 2: Negating a term
query = {
    "I need to buy a communications device, what should I get?": 1.0,
    "The device should work like an intelligent computer.": -0.3,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("\nQuery 2:")
pprint.pprint(results)
```

### Creating and Searching Indexes with Multimodal Combination Fields

Combine text and images in a single field and specify weights:

```python
import marqo
import pprint

mq = marqo.Client(url="http://localhost:8882")

settings = {"treat_urls_and_pointers_as_images": True, "model": "ViT-L/14"}

mq.create_index("my-first-multimodal-index", **settings)

mq.index("my-first-multimodal-index").add_documents(
    [
        {
            "Title": "Flying Plane",
            "caption": "An image of a passenger plane flying in front of the moon.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image2.jpg",
        },
        {
            "Title": "Red Bus",
            "caption": "A red double decker London bus traveling to Aldwych",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image4.jpg",
        },
        {
            "Title": "Horse Jumping",
            "caption": "A person riding a horse over a jump in a competition.",
            "image": "https://raw.githubusercontent.com/marqo-ai/marqo/mainline/examples/ImageSearchGuide/data/image1.jpg",
        },
    ],
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7
            }
        }
    },
    tensor_fields=["captioned_image"]
)

# Example 1: Text query
results = mq.index("my-first-multimodal-index").search(
    q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes."
)

print("Query 1:")
pprint.pprint(results)

# Example 2: Weighted query
results = mq.index("my-first-multimodal-index").search(
    q={
        "What are some vehicles and modes of transport?": 1.0,
        "Aeroplanes and other things that fly": -1.0
    },
)
print("\nQuery 2:")
pprint.pprint(results)

# Example 3: Negation
results = mq.index("my-first-multimodal-index").search(
    q={"Animals of the Perissodactyla order": -1.0}
)
print("\nQuery 3:")
pprint.pprint(results)
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

Marqo offers Kubernetes templates for deployment on your chosen cloud provider.  This allows for scalable clusters with replicas, storage shards, and inference nodes: [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)

For a fully managed solution, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Access the comprehensive Marqo documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Note

Do not run other applications on Marqo's Vespa cluster, as Marqo automatically modifies cluster settings.

## Contributors

Marqo is a community-driven project.  We welcome your contributions!  See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the environment: `source ./venv/bin/activate`.
3.  Install requirements: `pip install -r requirements.txt`.
4.  Run tests using `tox` (after navigating to the project directory).
5.  If you update dependencies, delete the `.tox` directory and rerun `tox`.

## Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with a linked GitHub issue.

## Support

*   Ask questions and share ideas on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack) for discussions.