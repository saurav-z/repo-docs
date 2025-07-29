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

Marqo is an open-source vector search engine that simplifies building powerful search applications for both text and images, offering vector generation, storage, and retrieval through a single API.  [Get started with Marqo on GitHub!](https://github.com/marqo-ai/marqo)

**Key Features:**

*   **🤖 State-of-the-art Embeddings:** Utilize cutting-edge machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU acceleration.
*   **⚡ Blazing Performance:** Benefit from in-memory HNSW indexes for rapid search speeds.
    *   Scale effortlessly to hundreds of millions of documents with index sharding.
    *   Enjoy asynchronous and non-blocking data upload and search operations.
*   **🌌 Documents-In, Documents-Out:** Simplify your workflow with integrated vector generation, storage, and retrieval.
    *   Easily build search, entity resolution, and data exploration applications for text and images.
    *   Construct complex semantic queries using weighted search terms.
    *   Filter search results using Marqo's query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, supporting a variety of data types (booleans, integers, keywords, etc.)
*   **🍱 Managed Cloud Option:** Leverage a fully managed cloud offering for optimized deployment.
    *   Scale inference with ease.
    *   Benefit from high availability and 24/7 support.
    *   Includes access control features.
    *   Explore Marqo Cloud [here](https://www.marqo.ai/cloud).

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Use Marqo as your Document Store for Haystack pipelines, including retrieval-augmentation and question answering.
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Access scalable search within LLM-based agents using the MarqoVectorStoreDriver.
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Integrate Marqo for vector search in your LangChain applications.
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Leverage Marqo for Hamilton LLM applications.

## Learn More

| Feature                                                       | Description                                                                                                                                                                                |
| ------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| 📗 [Quick Start](#getting-started)                            | Build your first application with Marqo in under 5 minutes.                                                                                                                                 |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Learn how to build advanced image search capabilities.                                                                                                          |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Learn how to build a multilingual database using Marqo.                                                                                                       |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Discover how to make GPT a subject matter expert with Marqo.                                                                                                              |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Explore how to combine stable diffusion with semantic search.                                                                                                 |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                                             |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Build advanced image search with Marqo to find and remove content.                                                                                        |
| ☁️ [Getting Started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Learn how to set up and start building your first application with Marqo Cloud.                                                                                  |
| 👗 [Marqo for E-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Explore a web application with frontend and backend using Python, Flask, ReactJS, and Typescript that make requests to your Marqo cloud API.                   |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chat bot application using Marqo and OpenAI's ChatGPT API.                                                                                               |
| 🦾 [Core Features](#-core-features)                              | Review Marqo's comprehensive core features.                                                                                                                                              |

## Getting Started

1.  **Prerequisites:** Docker is required. Install it from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage allocated.
2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your `marqo` container crashes, increase Docker's memory allocation to at least 6GB (8GB recommended).*
3.  **Install the Marqo client:**
    ```bash
    pip install marqo
    ```
4.  **Start indexing and searching!**
    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2") # Specify model
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

    *   `mq` is the Marqo client.
    *   `create_index()` creates a new index. You can specify the model (e.g., `"hf/e5-base-v2"`). Experiment to find the best model for your use case.
    *   `add_documents()` adds documents for indexing. `tensor_fields` specifies fields to index as vector collections.
    *   You can set a document's ID with the `_id` field. Marqo generates one if not provided.

    Example Results:

    ```python
    import pprint
    pprint.pprint(results)
    ```

    *Each hit shows a document matching the search query, ordered by relevance.*  `_highlights` indicates the matched portion, and `_score` is the relevancy score.

## Other Basic Operations

### Get Document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*   *Note:* Adding a document with the same `_id` updates the existing document.

### Get Index Stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi Modal and Cross Modal Search

Marqo supports image and text search using CLIP models.

1.  **Create an index with CLIP configuration:**

    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```

2.  **Add images to documents:**

    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```

3.  **Search images using text:**

    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```

### Searching Using an Image

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching with Weighted Queries

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

# Example of weighted and negated queries
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

# Example of weighted and negated queries
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

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo in Production

Marqo offers robust deployment options:

*   **Kubernetes:**  Use our Kubernetes templates for deployment on your cloud provider.  Find the repository [here](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed experience, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Warning

Do not run other applications on Marqo's Vespa cluster, as Marqo manages and adapts the cluster's settings automatically.

## Contributors

Marqo is a community project. We welcome contributions!  Please read the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## Dev Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox` (from this directory).
5.  If updating dependencies, delete the `.tox` directory and rerun `tox`.

## Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with a linked GitHub issue.

## Support

*   Join our [Discourse forum](https://community.marqo.ai) to ask questions and share your projects.
*   Connect with the community on our [Slack community](https://bit.ly/marqo-community-slack).