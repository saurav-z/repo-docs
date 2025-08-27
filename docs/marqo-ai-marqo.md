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

**Tired of complex vector search setups? Marqo provides an end-to-end solution for vector generation, storage, and retrieval with a single API.** Explore the [Marqo Repository](https://github.com/marqo-ai/marqo) to learn more.

## Key Features

*   **🤖 Advanced Embeddings:**
    *   Utilize state-of-the-art machine learning models from Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU acceleration.

*   **⚡ High Performance:**
    *   Leverages in-memory HNSW indexes for blazing-fast search speeds.
    *   Scales to hundreds of millions of documents with horizontal index sharding.
    *   Offers async and non-blocking data upload and search capabilities.

*   **🌌 Documents-in-Documents-Out:**
    *   Provides vector generation, storage, and retrieval out-of-the-box.
    *   Build search, entity resolution, and data exploration applications using your text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo's powerful query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, supporting various datatypes like bools, ints, and keywords.

*   **🍱 Managed Cloud (Marqo Cloud):**
    *   Optimized deployment with low latency.
    *   Scale inference with ease.
    *   High availability with 24/7 support.
    *   Access control features.
    *   Learn more [here](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks, with more integrations continually being added:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More About Marqo

| Topic                                                                                                 | Description                                                                                                        |
| :---------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick Start](#getting-started)                                                                     | Get started with Marqo in under 5 minutes.                                                                         |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Build advanced image search applications with Marqo.                                                           |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Build multilingual databases using Marqo.                                                                         |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Enhance GPT with Marqo to create a knowledge base and improve question-answering.                                     |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine stable diffusion with semantic search to generate and categorize large datasets of images.                  |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                                | Add diarization and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Utilize Marqo to build applications to identify and remove undesired content.                                 |
| ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)           | Learn to set up and run Marqo Cloud and start building applications.                                                 |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Example of how to build a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)         | Step-by-step guide to create a chatbot application using Marqo and OpenAI's ChatGPT API.                                  |
| 🦾 [Core Features](#-core-features)                                                                       | Detailed description of Marqo's core capabilities.                                                                    |

## Getting Started

Follow these steps to begin using Marqo:

1.  **Install Docker:** Download and install Docker from the [official Docker website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage allocated.
2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your `marqo` container is frequently crashing, increase Docker's memory allocation in the settings to at least 6GB (8GB recommended).*
3.  **Install the Marqo Client:**
    ```bash
    pip install marqo
    ```
4.  **Start Indexing and Searching:**

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2")  # Optional: Specify the model

    mq.index("my-first-index").add_documents([
        {
            "Title": "The Travels of Marco Polo",
            "Description": "A 13th-century travelogue describing Polo's travels"
        },
        {
            "Title": "Extravehicular Mobility Unit (EMU)",
            "Description": "The EMU is a spacesuit that provides environmental protection, mobility, life support, and communications for astronauts",
            "_id": "article_591"
        }],
        tensor_fields=["Description"]
    )

    results = mq.index("my-first-index").search(
        q="What is the best outfit to wear on the moon?"
    )
    ```
    *   `mq` initializes the Marqo client.
    *   `create_index()` creates a new index (optionally specify a model).
    *   `add_documents()` adds documents for indexing. `tensor_fields` specifies fields to be indexed as vectors.
    *   You can set a document's ID with the `_id` field. Marqo will generate one if not provided.

    View the results:

    ```python
    import pprint
    pprint.pprint(results)
    ```

    *   The `hits` list displays matching documents, ordered by relevance.
    *   `_highlights` shows the parts of documents that best matched the query.

## Other Basic Operations

### Get Document

Retrieve a document by its ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```
Note that by adding the document using ```add_documents``` again using the same ```_id``` will cause a document to be updated.

### Get Index Stats

Retrieve index information.

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

Perform a keyword search.

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi-Modal and Cross-Modal Search

Leverage CLIP models for image and text search. **Image URLs are treated as strings unless multi-modal search is configured.** Create an index with a CLIP configuration:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Add images within documents:

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

Search using an image URL:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching Using Weights in Queries

Use queries as dictionaries with weightings for advanced queries:

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

Combine text and images for single-vector representation:

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

Remove documents by ID.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

Delete an index.

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo in Production

Deploy Marqo on your infrastructure with Kubernetes using the templates available at [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes). For a fully managed experience, use [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

Access comprehensive documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Notes

*   Do not run other applications on Marqo's Vespa cluster, as Marqo manages and adapts the cluster's settings automatically.

## Contributing

Marqo is a community-driven project.  We welcome your contributions! See [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the virtual environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests: `tox` (after navigating to the directory)
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite using the command `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   Engage with the community on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack) for real-time discussions.