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

Marqo is an end-to-end vector search engine that simplifies indexing, storing, and retrieving text and images, all through a single, easy-to-use API.  [Get started with Marqo on Github](https://github.com/marqo-ai/marqo)!

**Key Features:**

*   🤖 **State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, and OpenAI. Bring your own or choose from pre-configured models with CPU and GPU support.
*   ⚡ **High Performance:** Benefit from in-memory HNSW indexes for blazing-fast search speeds, horizontal index sharding for scaling to hundreds of millions of documents, and async/non-blocking data upload and search.
*   🌌 **Documents-in, Documents-out:** Simplify your workflow with built-in vector generation, storage, and retrieval for both text and images. Build complex semantic queries, filter results with a query DSL, and store unstructured data with metadata.
*   🍱 **Managed Cloud (Marqo Cloud):** Experience low-latency optimized deployment, effortless inference scaling, high availability, 24/7 support, and access control. Learn more [here](https://www.marqo.ai/cloud).

## Core Features (Expanded)

*   **Flexible Embedding Models**: Easily integrate various embedding models, including those from Hugging Face, OpenAI, and more, with options for both CPU and GPU acceleration.
*   **Optimized Indexing**: Utilizes in-memory HNSW indexes for efficient and rapid search results, allowing for quick retrieval of relevant documents.
*   **Scalable Architecture**: Built to handle large datasets, with support for horizontal index sharding, ensuring performance even with hundreds of millions of documents.
*   **Unified Data Handling**: Supports indexing and searching of text and images, simplifying multimodal search applications with a "documents-in, documents-out" approach.
*   **Advanced Querying**: Offers the ability to combine weighted search terms, enabling nuanced semantic queries, and allows for filtering results using a query DSL.

## Integrations

Marqo seamlessly integrates with popular AI and data processing frameworks:

*   💙 **Haystack:** Use Marqo as your document store for building NLP pipelines, including retrieval-augmented generation, question answering, and document search. (See [Haystack Integration](https://haystack.deepset.ai/integrations/marqo-document-store))
*   🛹 **Griptape:** Empower LLM-based agents with scalable search capabilities using MarqoVectorStoreDriver.
*   🦜🔗 **Langchain:** Integrate Marqo into LangChain applications to enhance vector search within retrieval QA and conversational retrieval QA chains.
*   ⋙ **Hamilton:** Leverage Marqo for your Hamilton LLM applications.

## Learn More

*   📗 **Quick Start:** Build your first Marqo application in under 5 minutes.
*   🖼️ **Marqo for Image Data:** Explore advanced image search capabilities.
*   📚 **Marqo for Text:** Discover how to build multilingual databases with Marqo.
*   🔮 **Integrating Marqo with GPT:** Learn how to use Marqo as a knowledge base to enhance GPT's performance.
*   🎨 **Marqo for Creative AI:** Combine Stable Diffusion with semantic search to generate and categorize images.
*   🔊 **Marqo and Speech Data:** Add diarization and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   🚫 **Marqo for Content Moderation:** Find and remove content with Marqo's advanced image search capabilities.
*   ☁️ **Getting Started with Marqo Cloud:** Walkthrough of the setup and usage of Marqo Cloud.
*   👗 **Marqo for E-commerce:** Build e-commerce applications.
*   🤖 **Marqo Chatbot:** Create chatbot applications with Marqo and OpenAI's ChatGPT API.
*   🦾 **Core Features:** Read more about Marqo's main features.

## Getting Started

Follow these steps to quickly get started with Marqo:

1.  **Install Docker:**  Ensure you have Docker installed.  See the [Docker Official website](https://docs.docker.com/get-docker/). Ensure that docker has at least 8GB memory and 50GB storage.
2.  **Run Marqo using Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

    *Note:  If you're running into issues, make sure Docker has at least 8GB of memory allocated.*

3.  **Install the Marqo Client:**

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
    *   `create_index()` creates a new index with default settings. You have the option to specify what model to use. For example, `mq.create_index("my-first-index", model="hf/e5-base-v2")` will create an index with the default text model `hf/e5-base-v2`.
    *   `add_documents()` takes a list of documents, represented as python dicts for indexing. `tensor_fields` refers to the fields that will be indexed as vector collections and made searchable.
    *   You can optionally set a document's ID with the special `_id` field. Otherwise, Marqo will generate one.

    ```python
    import pprint
    pprint.pprint(results)
    ```
    - Each hit corresponds to a document that matched the search query.
    - They are ordered from most to least matching.
    - `limit` is the maximum number of hits to be returned. This can be set as a parameter during search.
    - Each hit has a `_highlights` field. This was the part of the document that matched the query the best.

## Other Basic Operations

### Get Document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*Note that by adding the document using ```add_documents``` again using the same ```_id``` will cause a document to be updated.*

### Get Index Stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi Modal and Cross Modal Search

To enable image and text search, configure a CLIP model:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Add images to documents:

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

### Searching using an image

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching using Weights in Queries

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

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

```python
results = mq.index("my-first-index").delete()
```

## Running Marqo in Production

Marqo provides Kubernetes templates for easy deployment. For a fully managed cloud service, check out [Marqo Cloud](https://cloud.marqo.ai).

## Documentation

Find detailed documentation on the [Marqo Documentation Website](https://docs.marqo.ai/).

## Important Note

Do not run other applications on Marqo's Vespa cluster, as Marqo automatically manages the cluster settings.

## Contributing

Marqo thrives on community contributions!  See our [CONTRIBUTING.md](./CONTRIBUTING.md) for guidelines.

## Development Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate:  `source ./venv/bin/activate`
3.  Install requirements:  `pip install -r requirements.txt`
4.  Run tests: `tox`

## Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.

## Support

*   [Discourse Forum](https://community.marqo.ai): Ask questions and share your work.
*   [Slack Community](https://bit.ly/marqo-community-slack): Chat with the community.