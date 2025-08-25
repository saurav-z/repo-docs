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

**Marqo is a powerful, end-to-end vector search engine that lets you seamlessly search and retrieve information from both text and images with a single API.** ([Visit the original repo](https://github.com/marqo-ai/marqo))

## Key Features

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize cutting-edge machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own custom embeddings.
    *   Supports both CPU and GPU acceleration for optimal performance.
*   **⚡ High Performance:**
    *   Leverages in-memory HNSW indexes for blazing-fast search speeds.
    *   Scales to indexes with hundreds of millions of documents using horizontal index sharding.
    *   Offers asynchronous and non-blocking data upload and search operations.
*   **🌌 Documents-In-Documents-Out:**
    *   Handles vector generation, storage, and retrieval out of the box.
    *   Build search, entity resolution, and data exploration applications using both text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo's flexible query DSL.
    *   Store both unstructured data and semi-structured metadata together in documents, using a wide range of supported data types like booleans, integers, and keywords.
*   **🍱 Managed Cloud Option:**
    *   Low-latency, optimized deployment of Marqo in the cloud.
    *   Effortlessly scale inference with a single click.
    *   Provides high availability and 24/7 support.
    *   Includes access control features for enhanced security.
    *   Learn more about Marqo Cloud [here](https://www.marqo.ai/cloud).

## Integrations

Marqo integrates with popular AI and data processing frameworks, with more integrations constantly being added:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack):** Use Marqo as your Document Store for Haystack pipelines like retrieval-augmentation and question answering.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape):** Give your LLM-based agents access to scalable search using your own data through the MarqoVectorStoreDriver.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain):** Integrate Marqo for vector search within your LangChain applications.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/):** Leverage Marqo for Hamilton LLM applications.

## Learn More About Marqo

| Feature                                              | Description                                                                                                                                                                             |
| ---------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#getting-started)                   | Build your first application with Marqo in under 5 minutes.                                                                                                                            |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Build advanced image search with Marqo.                                                                                                                                  |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Build a multilingual database in Marqo.                                                                                                                              |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Make GPT a subject matter expert by using Marqo as a knowledge base.                                                                                                                 |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine Stable Diffusion with semantic search to generate and categorize 100k images of hotdogs.                                                                         |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarization and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                                                |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Build advanced image search with Marqo to find and remove content.                                                                                                        |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Get started with Marqo Cloud from your first login to building your first application.                                                                                                |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chat bot application using Marqo and OpenAI's ChatGPT API.                                                                                                   |
| 🦾 [Core Features](#-core-features)                  | Marqo's core features.                                                                                                                                                                  |

## Getting Started

1.  **Prerequisites:** Marqo requires Docker. Install it from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB memory (16GB recommended) and 50GB storage allocated.

2.  **Run Marqo using Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```

    **Note:** If your `marqo` container keeps getting killed, increase the memory limit for Docker to at least 6GB (8GB recommended) in your Docker settings.

3.  **Install the Marqo client:**

    ```bash
    pip install marqo
    ```

4.  **Start Indexing and Searching:**

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

    *   `mq` is the client that wraps the `marqo` API.
    *   `create_index()` creates a new index with default settings. You can specify the model (e.g., `hf/e5-base-v2`).
    *   `add_documents()` takes a list of documents, represented as Python dicts, for indexing. `tensor_fields` specifies which fields to index as vector collections.
    *   You can optionally set a document's ID with the `_id` field; otherwise, Marqo will generate one.

    Let's examine the search results:

    ```python
    import pprint
    pprint.pprint(results)
    ```

    *   Each "hit" is a matching document, ordered by relevance.
    *   `_highlights` show the parts of the document that best matched the query.
    *   `_score` indicates the relevance.

## Other Basic Operations

### Get Document

Retrieve a document by its ID:

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*Adding a document using `add_documents` with the same `_id` will update that existing document.*

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

Marqo supports image and text search using CLIP models.  First, create an index with a CLIP configuration:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Add images within documents using URLs or local file paths:

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

Search by providing an image link:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching Using Weights in Queries

Use dictionaries for queries, with keys as queries and values as weights:

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

*   **Kubernetes:** Deploy Marqo using our Kubernetes templates.  Find them at: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** For a fully managed cloud service, sign up at: [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/)

## Important Note

Avoid running other applications on Marqo's Vespa cluster, as Marqo automatically adjusts cluster settings.

## Contributing

Marqo is a community project. We welcome your contributions!  Read our [CONTRIBUTING guidelines](./CONTRIBUTING.md) to get started.

## Dev Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox` (from within the root directory)
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite using the command `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   **Community Forum:** Ask questions and share your creations at our [Discourse forum](https://community.marqo.ai).
*   **Slack:** Join our [Slack community](https://bit.ly/marqo-community-slack) to chat with other community members.