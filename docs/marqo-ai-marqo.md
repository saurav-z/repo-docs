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

**Marqo simplifies vector search by providing an end-to-end solution for both text and image indexing, storage, and retrieval through a single API.** [Explore the Marqo repository on GitHub](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU for efficient processing.
*   **High Performance:**
    *   Leverages in-memory HNSW indexes for cutting-edge search speeds.
    *   Scales to handle indexes with hundreds of millions of documents using horizontal index sharding.
    *   Offers asynchronous and non-blocking data upload and search for optimal performance.
*   **Documents-in, Documents-out:**
    *   Provides out-of-the-box vector generation, storage, and retrieval.
    *   Build applications for search, entity resolution, and data exploration using both text and images.
    *   Supports complex semantic queries with weighted search terms.
    *   Includes a query DSL for filtering search results.
    *   Stores both unstructured data and semi-structured metadata within documents, with support for various datatypes.
*   **Managed Cloud Option:**
    *   Optimized deployment of Marqo with low latency.
    *   Scale inference with ease.
    *   High availability.
    *   24/7 support.
    *   Access control.
    *   Learn more [here](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More about Marqo

|                                                             |                                                                                                                                                                                                          |
| ----------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)                         | Build your first application with Marqo in under 5 minutes.                                                                                                                                                 |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo.                                                                                                      |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo.                                                                                                      |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.                                                                                                                              |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.                                             |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                  | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.       |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content.                                                                               |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo                                          |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript.                                                                  |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API.                                                                   |
| 🦾 [Core Features](#-core-features)        | Marqo's core features.                                                                                                                                                                                                      |

## Getting Started

Follow these steps to quickly get up and running with Marqo:

1.  **Prerequisites:** Marqo requires Docker.  Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB memory and 50GB storage allocated.

2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

    *Note: If your `marqo` container is getting killed, increase the Docker memory limit in Docker settings to at least 6GB (8GB recommended).*

3.  **Install the Marqo client:**

    ```bash
    pip install marqo
    ```

4.  **Start indexing and searching!** Here's a simple example:

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

    *   `mq` is the client for interacting with the `marqo` API.
    *   `create_index()` creates a new index.  Specify the model with the `model` parameter (e.g.,  `mq.create_index("my-first-index", model="hf/e5-base-v2")`).  Experiment to optimize retrieval for your use case.  See [here](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) for a full list of models.
    *   `add_documents()` adds documents for indexing, represented as Python dictionaries.  `tensor_fields` specifies the fields to be indexed as vector collections.
    *   You can optionally set a document's ID using the `_id` field.

    Let's examine the results:

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

    *   Each hit represents a matching document.
    *   Results are ordered from most to least relevant.
    *   `limit` controls the maximum number of hits returned.
    *   `_highlights` shows the parts of the document that best matched the query.

## Other Basic Operations

### Get Document

Retrieve a document by its ID:

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*Note: Using `add_documents` again with the same `_id` will update the existing document.*

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

Marqo enables image and text search with CLIP models from Hugging Face.  **Image URLs are treated as strings unless you configure multimodal search.**  To start indexing images:

1.  **Create an index with a CLIP configuration:**

    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```

2.  **Add images to documents (using URLs or local file paths):**

    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```

3.  **Search the image field using text:**

    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```

### Searching using an image
Searching using an image can be achieved by providing the image link.

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching using weights in queries
Queries can also be provided as dictionaries where each key is a query and their corresponding values are weights. This allows for more advanced queries consisting of multiple components with weightings towards or against them, queries can have negations via negative weighting.

The example below shows the application of this to a scenario where a user may want to ask a question but also negate results that match a certain semantic criterion. 

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

### Creating and searching indexes with multimodal combination fields
Marqo lets you have indexes with multimodal combination fields. Multimodal combination fields can combine text and images into one field. This allows scoring of documents across the combined text and image fields together. It also allows for a single vector representation instead of needing many which saves on storage. The relative weighting of each component can be set per document.

The example below demonstrates this with retrieval of caption and image pairs using multiple types of queries.

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

Marqo offers several options for production deployment:

*   **Kubernetes:**  We provide Kubernetes templates for deployment on any cloud provider. See [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:**  For a fully managed cloud service, sign up at [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

Find the full Marqo documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Notes

*   Do not run other applications on Marqo's Vespa cluster. Marqo automatically manages and adapts its settings.

## Contributing

Marqo is a community-driven project. We welcome contributions! Please review our [contribution guidelines](./CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`.
2.  Activate the virtual environment: `source ./venv/bin/activate`.
3.  Install dependencies: `pip install -r requirements.txt`.
4.  Run tests:  Use `tox` in the root directory.
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite using the `tox` command.
2.  Create a pull request with an associated GitHub issue.

## Support

*   **Discourse Forum:** Ask questions and share your ideas on our [Discourse forum](https://community.marqo.ai).
*   **Slack Community:** Join our [Slack community](https://bit.ly/marqo-community-slack) for discussions and support.