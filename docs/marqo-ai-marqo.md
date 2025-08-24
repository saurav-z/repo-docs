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

**Marqo empowers developers to build powerful vector search applications with an intuitive, end-to-end solution for text and image search.** ([See the original repo](https://github.com/marqo-ai/marqo))

### Key Features

*   **🤖 Cutting-Edge Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more. Easily start with pre-configured models or bring your own, with both CPU and GPU support.
*   **⚡ High Performance:** Experience lightning-fast search speeds with embeddings stored in in-memory HNSW indexes. Scale to hundreds of millions of documents with horizontal index sharding, and benefit from async and non-blocking data upload and search capabilities.
*   **🌌 Documents-In-Documents-Out:**  Simplify vector generation, storage, and retrieval with an out-of-the-box solution. Build robust search, entity resolution, and data exploration applications using your text and images. Craft sophisticated semantic queries and filter results with Marqo’s query DSL. Store unstructured data and metadata together using various data types like booleans, integers, and keywords.
*   **🍱 Managed Cloud:** Take advantage of a low-latency optimized deployment of Marqo, along with effortless inference scaling. Enjoy high availability, 24/7 support, and access control features. Learn more about Marqo Cloud [here](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Search:** Marqo supports multimodal search capabilities. Seamlessly search images and text with CLIP models.
*   **⚖️ Weighted Queries:** Construct advanced queries with weights for enhanced precision.
*   **🧑‍💻 Easy to Use:** Built for developers, easily set up and build your first application within minutes.

### Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks, making it easier than ever to incorporate vector search into your projects:

*   💙 [Haystack](https://github.com/deepset-ai/haystack)
*   🛹 [Griptape](https://github.com/griptape-ai/griptape)
*   🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)
*   ⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)

### Learn More

| Topic                                                                                                                              | Description                                                                                                                                                |
| ----------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick Start](#getting-started)                                                                                                  | Get started and build your first application with Marqo in under 5 minutes.                                                                            |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)      | Explore advanced image search capabilities with Marqo.                                                                                                  |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)      | Discover how to build multilingual databases with Marqo.                                                                                                |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Learn how to use Marqo as a knowledge base to enhance GPT.                                                                                               |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine stable diffusion with semantic search to generate and categorize images.                                                                       |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                                                              | Add diarization and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                       |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)       | Build advanced image search with Marqo to find and remove content.                                                                                      |
| ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)                                    |  Start with Marqo Cloud, from your first login through to building your first application with Marqo.                                                      |
| 👗 [Marqo for E-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)            | Build a web application with frontend and backend using Python, Flask, ReactJS, and Typescript.                                                        |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)                                | Build a chat bot application using Marqo and OpenAI's ChatGPT API.                                                                                      |
| 🦾 [Core Features](#-core-features)                                                                                              | Explore the core features of Marqo.                                                                                                                          |

## Getting Started

Follow these steps to quickly get started with Marqo:

1.  **Docker Setup:**  Ensure you have Docker installed.  [Docker Official Website](https://docs.docker.com/get-docker/). Allocate at least 8GB memory and 50GB storage to Docker.
2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your `marqo` container is getting killed, increase the memory limit for Docker to at least 6GB (8GB recommended) in your Docker settings.*
3.  **Install Marqo Client:**
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
    *   `mq`: The Marqo client.
    *   `create_index()`: Creates a new index.  Specify the model (e.g., `"hf/e5-base-v2"`). See [Dense Retrieval Models](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) for the full list.
    *   `add_documents()`: Adds documents for indexing (as Python dictionaries).  `tensor_fields` specifies fields for vector indexing.
    *   `_id`:  Optional field for document ID; otherwise, Marqo generates one.

    Example results:
    ```python
    import pprint
    pprint.pprint(results)

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
    *   `hits`: Matching documents.
    *   `_highlights`: Parts of the document that matched the query.
    *   `_score`: Relevance score.

## Other Basic Operations

### Get Document

Retrieve a document by ID:

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*Adding a document with an existing `_id` will update the existing document.*

### Get Index Stats

Get information about an index:

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

Perform a keyword search:

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multimodal and Cross-Modal Search

*   **Configure Multimodal Index:** Enable image search with CLIP models:
    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```
*   **Add Images to Documents:**
    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```
*   **Search with Text:**
    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```
### Searching Using an Image

Searching using an image can be achieved by providing the image link.

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching Using Weights in Queries

Apply weights to search terms for more precise results:

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

# initially we ask for a type of communications device which is popular in the 21st century
query = {
    # a weighting of 1.1 gives this query slightly more importance
    "I need to buy a communications device, what should I get?": 1.1,
    # a weighting of 1 gives this query a neutral importance
    # this will lead to 'Smartphone' being the top result
    "The device should work like an intelligent computer.": 1.0,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("Query 1:")
pprint.pprint(results)

# now we ask for a type of communications which predates the 21st century
query = {
    # a weighting of 1 gives this query a neutral importance
    "I need to buy a communications device, what should I get?": 1.0,
    # a weighting of -1 gives this query a negation effect
    # this will lead to 'Telephone' being the top result
    "The device should work like an intelligent computer.": -0.3,
}

results = mq.index("my-weighted-query-index").search(q=query)

print("\nQuery 2:")
pprint.pprint(results)
```

### Creating and Searching Indexes with Multimodal Combination Fields

Combine text and images in a single field for richer results:

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
    # Create the mappings, here we define our captioned_image mapping 
    # which weights the image more heavily than the caption - these pairs 
    # will be represented by a single vector in the index
    mappings={
        "captioned_image": {
            "type": "multimodal_combination",
            "weights": {
                "caption": 0.3,
                "image": 0.7
            }
        }
    },
    # We specify which fields to create vectors for. 
    # Note that captioned_image is treated as a single field.
    tensor_fields=["captioned_image"]
)

# Search this index with a simple text query
results = mq.index("my-first-multimodal-index").search(
    q="Give me some images of vehicles and modes of transport. I am especially interested in air travel and commercial aeroplanes."
)

print("Query 1:")
pprint.pprint(results)

# search the index with a query that uses weighted components
results = mq.index("my-first-multimodal-index").search(
    q={
        "What are some vehicles and modes of transport?": 1.0,
        "Aeroplanes and other things that fly": -1.0
    },
)
print("\nQuery 2:")
pprint.pprint(results)

results = mq.index("my-first-multimodal-index").search(
    q={"Animals of the Perissodactyla order": -1.0}
)
print("\nQuery 3:")
pprint.pprint(results)
```

### Delete Documents

Delete documents by ID:

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

Delete an index:

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo in Production

*   **Kubernetes:** Deploy Marqo using provided Kubernetes templates: [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** Get a fully managed cloud service: [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

Find comprehensive documentation at: [Marqo Documentation](https://docs.marqo.ai/).

## Important Note

Do not run other applications on Marqo's Vespa cluster; Marqo automatically manages and adapts its settings.

## Contributing

Marqo is a community project.  See the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## Development Setup

1.  Create and activate a virtual environment:
    ```bash
    python -m venv ./venv
    source ./venv/bin/activate
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run tests:
    ```bash
    tox
    ```
4.  If you update dependencies, delete the `.tox` directory and rerun tests.

## Merge Instructions

1.  Run the full test suite (using `tox`).
2.  Create a pull request with a linked GitHub issue.

## Support

*   Ask questions and share your creations on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack).