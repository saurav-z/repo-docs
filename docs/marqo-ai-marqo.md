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

**Marqo is a powerful, end-to-end vector search engine that simplifies the process of building semantic search applications for both text and images, offering a complete solution with built-in vector generation, storage, and retrieval.**  [See the original repo](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own custom embeddings.
    *   Supports both CPU and GPU for optimal performance.
*   **⚡ High Performance and Scalability:**
    *   Leverages in-memory HNSW indexes for blazing-fast search speeds.
    *   Scale to massive indexes with hundreds of millions of documents using horizontal sharding.
    *   Benefit from async and non-blocking data upload and search operations.
*   **🌌 Documents-in-Documents-Out:**
    *   Vector generation, storage, and retrieval are provided out-of-the-box for a streamlined experience.
    *   Build powerful search applications, entity resolution, and data exploration using your text and images.
    *   Create complex semantic queries by combining weighted search terms for precision.
    *   Refine your search results using Marqo's flexible query DSL.
    *   Store both unstructured data and structured metadata together in documents, supporting various datatypes (booleans, integers, keywords, etc.).
*   **🍱 Managed Cloud Option:**
    *   Benefit from a low-latency, optimized deployment of Marqo in the cloud.
    *   Scale inference with ease at the click of a button.
    *   Enjoy high availability and 24/7 support.
    *   Leverage built-in access control for enhanced security.
    *   Learn more about Marqo Cloud [here](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks, and more integrations are continually being developed:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Use Marqo as your Document Store for Haystack pipelines.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Access scalable search for your LLM-based agents.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Integrate with LangChain applications through a vector store implementation.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Leverage Marqo for Hamilton LLM applications.

## Explore Marqo's Capabilities

|  |  |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick Start](#getting-started) | Build your first application with Marqo in minutes. |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Explore advanced image search functionalities within Marqo. |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Learn how to build a multilingual database using Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Augment GPT with Marqo for context-aware question answering. |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine stable diffusion with semantic search to generate and categorize images. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) |  Preprocess audio for question answering with Marqo and ChatGPT. |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Build advanced image search to detect and remove unwanted content. |
| ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Guide on setting up and running with Marqo Cloud. |
| 👗 [Marqo for E-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) |  Web application with frontend and backend, using Python, Flask, ReactJS, and Typescript for e-commerce. |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chatbot with Marqo and OpenAI's ChatGPT API. |
| 🦾 [Core Features](#-core-features) |  Explore Marqo's extensive core feature set. |

## Getting Started

Follow these steps to get started with Marqo:

1.  **Install Docker:**  Marqo requires Docker. Install it from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage allocated.
2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    **Note:** If your `marqo` container crashes, allocate at least 6GB (8GB recommended) memory to Docker.
3.  **Install the Marqo Client:**

    ```bash
    pip install marqo
    ```
4.  **Start Indexing and Searching!** Here's a simple example:

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
    *   `create_index()` creates a new index (you can specify the model).
    *   `add_documents()` adds documents to the index (use `tensor_fields` to specify fields for vector indexing).
    *   You can optionally set a document's ID using the `_id` field; otherwise, Marqo will generate one.

    View the search results:

    ```python
    import pprint
    pprint.pprint(results)
    ```

    Output example:

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

    *   Each hit represents a matching document, ordered by relevance (`_score`).
    *   `_highlights` shows the matched portions of the document.

## Other Basic Operations

### Get Document

Retrieve a document by its ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

### Update documents

Add documents again using the same `_id` and update the document.

### Get Index Stats

Retrieve information about an index.

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

Perform a keyword search.

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi-Modal and Cross-Modal Search

Marqo enables image and text search using CLIP models from Hugging Face. To enable multi-modal search, create an index with a CLIP configuration:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Add images via URLs or local paths:

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

Search the image field with text:

```python
results = mq.index("my-multimodal-index").search('animal')
```

### Searching with an Image

Use an image URL for image-based search:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Weighted Queries

Use dictionaries to create complex queries with weighted terms:

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

### Multimodal Combination Fields

Combine text and images into a single field for combined scoring:

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

Delete documents by their IDs.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

Delete an index.

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo in Production

Marqo offers options for production deployments:

*   **Kubernetes:**  We support Kubernetes templates for deploying Marqo clusters on your preferred cloud provider.  See [marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed cloud service, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation for Marqo is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Notes

Do not run other applications on Marqo's Vespa cluster, as Marqo automatically adjusts the cluster's settings.

## Contributing

Marqo is a community project.  We welcome your contributions!  See the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the virtual environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests:  Run `tox` in this directory.
5.  Update dependencies: Delete the `.tox` directory and rerun `tox`.

## Merge Instructions

1.  Run the full test suite (using `tox`).
2.  Create a pull request with an attached GitHub issue.

## Support

*   Join the [Discourse forum](https://community.marqo.ai) to ask questions and share your projects.
*   Join our [Slack community](https://bit.ly/marqo-community-slack) to discuss ideas.