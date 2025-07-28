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

**Marqo empowers developers to build powerful search experiences with its all-in-one vector search engine, simplifying the process of generating, storing, and retrieving vectors for both text and image data.**  ([View the original repository](https://github.com/marqo-ai/marqo))

**Key Features:**

*   **🚀 Out-of-the-Box Vector Search:**  Handles vector generation, storage, and retrieval with a single API for text and images.
*   **✨ State-of-the-Art Embeddings:** Utilize the latest machine learning models from Hugging Face, OpenAI, and more, with support for both CPU and GPU.
*   **⚡ High-Performance Search:**  Experience cutting-edge search speeds with in-memory HNSW indexes, scalable to hundreds of millions of documents.
*   **🌌 Documents-in, Documents-Out:** Simplify your workflow with a streamlined "documents-in, documents-out" approach, handling preprocessing and metadata storage.
*   **🍱 Managed Cloud Option:** Leverage a fully managed cloud service with optimized deployment, scalability, high availability, and 24/7 support.
*   **🖼️ Multimodal Search:** Seamlessly integrate image and text search with CLIP models.
*   **🌐 Flexible Querying:** Combine weighted search terms, utilize a query DSL, and store unstructured data alongside metadata.
*   **🔗 Integrations:**  Integrates with popular AI and data processing frameworks like Haystack, Griptape, Langchain, and Hamilton.

## Why Choose Marqo?

Marqo simplifies vector search by providing an end-to-end solution.  It bundles essential components like ML model deployment, input preprocessing, and metadata management, allowing developers to focus on building innovative search applications.  No need to manage separate vector databases or embedding pipelines – Marqo handles it all.

## Getting Started - Quickstart Guide

**1. Docker Setup:**

*   Install Docker from the [Docker Official Website](https://docs.docker.com/get-docker/).  Ensure at least 8GB memory and 50GB storage is allocated in Docker settings.

**2. Run Marqo with Docker:**

```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
```

**3. Install the Marqo Client:**

```bash
pip install marqo
```

**4. Indexing and Searching Example:**

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

**5. Further operations**
*   Get document
```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```
*   Get index stats
```python
results = mq.index("my-first-index").get_stats()
```
*   Lexical search
```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```
*   Multimodal and cross modal search
```python
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

```python
results = mq.index("my-multimodal-index").search('animal')
```
*   Searching using an image
```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

*   Searching using weights in queries

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
*   Creating and searching indexes with multimodal combination fields
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
*   Delete documents
```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```
*   Delete index
```python
results = mq.index("my-first-index").delete()
```

## Production Deployment

Marqo supports Kubernetes templates for production deployments.

*   **Kubernetes:** Deploy scalable Marqo clusters with replicas, storage sharding, and inference nodes. [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)
*   **Marqo Cloud:** For a fully managed experience, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Note

Do not run other applications on Marqo's Vespa cluster as Marqo automatically modifies cluster settings.

## Contributing

Marqo welcomes community contributions!  Please review the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: `tox` (after navigating to the project directory)
5.  Update dependencies: Delete the `.tox` directory and rerun `tox` if dependencies are modified.

## Merge Instructions

1.  Run the full test suite (`tox`).
2.  Create a pull request with a linked GitHub issue.

## Support

*   **Discourse:** Get help and share your work on our [Discourse forum](https://community.marqo.ai).
*   **Slack:**  Join our [Slack community](https://bit.ly/marqo-community-slack) to chat with other community members.