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

# Marqo: The Open-Source, End-to-End Vector Search Engine for Text and Images

**Marqo simplifies vector search by providing a unified API for vector generation, storage, and retrieval, making it easy to build powerful search applications.** ([See the original repo](https://github.com/marqo-ai/marqo))

## Key Features:

*   **🤖 State-of-the-Art Embeddings:**
    *   Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.
    *   Choose from pre-configured models or bring your own.
    *   Supports both CPU and GPU acceleration.

*   **⚡ Blazing Performance:**
    *   Leverages in-memory HNSW indexes for cutting-edge search speeds.
    *   Scales to handle indexes with hundreds of millions of documents using horizontal index sharding.
    *   Offers asynchronous and non-blocking data upload and search operations.

*   **🌌 Documents-In-Documents-Out:**
    *   Provides out-of-the-box vector generation, storage, and retrieval capabilities.
    *   Build search, entity resolution, and data exploration applications with text and images.
    *   Create complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo's intuitive query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, utilizing various datatypes like booleans, integers, and keywords.

*   **🍱 Managed Cloud Option:**
    *   Optimized deployment of Marqo for low latency.
    *   Scale inference with the click of a button.
    *   High availability and 24/7 support.
    *   Access control for enhanced security.
    *   Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).

## Integrations

Marqo integrates with popular AI and data processing frameworks.

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**: Use Marqo as your Document Store for Haystack pipelines, enabling retrieval-augmentation, question answering, and document search.

*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**: Integrate Marqo with Griptape to leverage scalable search with your own data within LLM-based agents.

*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**: Utilize Marqo for LangChain applications, offering a vector search component that can plug into existing chains like Retrieval QA and Conversational Retrieval QA.

*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**: Integrate Marqo with Hamilton LLM applications.

## Learn More

| Topic                                                                 | Description                                                                                                                                         |
| --------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick Start](#getting-started)                                   | Build your first application with Marqo in under 5 minutes.                                                                                         |
| 🖼 [Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Explore advanced image search capabilities with Marqo.                                                                                                  |
| 📚 [Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Build a multilingual database using Marqo.                                                                                                             |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Enhance GPT's knowledge by using Marqo as a knowledge base.                                                                                                |
| 🎨 [Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combine Stable Diffusion with semantic search to generate and categorize a large image dataset.                                                       |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarization and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                 |
| 🚫 [Marqo for Content Moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Build advanced image search functionality to detect and remove inappropriate content.                                                                      |
| ☁️ [Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Learn how to get started with Marqo Cloud, from initial login to building your first application.                                                   |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | Learn how to build an e-commerce web application using Python, Flask, ReactJS, and TypeScript.                                                              |
| 🤖 [Marqo Chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | Build a chatbot application using Marqo and OpenAI's ChatGPT API.                                                                                        |
| 🦾 [Core Features](#-core-features)                                  | Review Marqo's core features.                                                                                                                        |

## Getting Started

Follow these steps to quickly get started with Marqo:

1.  **Docker Installation:** Ensure you have Docker installed ([Docker Official website](https://docs.docker.com/get-docker/)).  Allocate at least 8GB of memory and 50GB of storage to Docker.

2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```

    *Note: If the Marqo container is killed frequently, increase Docker's memory allocation (recommended at least 8GB) in your Docker settings.*

3.  **Install the Marqo Client:**

    ```bash
    pip install marqo
    ```

4.  **Index and Search:**  Here's a simple example:

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

    import pprint
    pprint.pp(results)
    ```

*   `mq` is the Marqo API client.
*   `create_index()` creates a new index.  You can specify a model (e.g., `hf/e5-base-v2`).
*   `add_documents()` adds documents to the index.  `tensor_fields` specifies which fields to vectorize for searching.

## Other Basic Operations

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

Enable image and text search with CLIP models:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Add images using URLs or local file paths:

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

Search using text:

```python
results = mq.index("my-multimodal-index").search('animal')
```

Search using an Image:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching with Weights in Queries

Use weighted queries for advanced search scenarios:

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

*   **Kubernetes:** Deploy Marqo on your cloud provider using provided Kubernetes templates:  [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed service, sign up at [Marqo Cloud](https://cloud.marqo.ai).

## Important Considerations

*   **Do not run other applications on Marqo's Vespa cluster,** as Marqo manages the settings automatically.

## Contributing

We welcome contributions!  See [CONTRIBUTING.md](CONTRIBUTING.md) to learn how to get involved.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the virtual environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests:  `tox` (after navigating to the directory).
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite: `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   **Discourse Forum:** Ask questions and share on the [Discourse forum](https://community.marqo.ai).
*   **Slack Community:** Join our [Slack community](https://bit.ly/marqo-community-slack).