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

**Marqo simplifies vector search with an end-to-end solution for generating, storing, and retrieving vectors for both text and images, all through a single, easy-to-use API.** ([View on GitHub](https://github.com/marqo-ai/marqo))

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.  Choose a pre-configured model or integrate your own, with both CPU and GPU support.
*   **⚡ High Performance:** Benefit from in-memory HNSW indexes for blazing-fast search speeds and scale to hundreds of millions of documents with horizontal index sharding, plus async, non-blocking data uploads and searches.
*   **🌌 Documents-in, Documents-out:** Vector generation, storage, and retrieval are handled out-of-the-box. Easily build search, entity resolution, and data exploration applications with your text and images.  Create complex semantic queries using weighted terms and filter results with Marqo's query DSL.
*   **🍱 Managed Cloud Option:** Enjoy a low-latency optimized deployment of Marqo, with scalable inference, high availability, 24/7 support, and access control.  Learn more about Marqo Cloud [here](https://www.marqo.ai/cloud).

## Core Features

*   **🤖 State of the art embeddings**
    *   Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more. 
    *   Start with a pre-configured model or bring your own.
    *   CPU and GPU support.

*   **⚡ Performance**
    *   Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
    *   Scale to hundred-million document indexes with horizontal index sharding.
    *   Async and non-blocking data upload and search.

*   **🌌 Documents-in-documents-out**
    *   Vector generation, storage, and retrieval are provided out of the box.
    *   Build search, entity resolution, and data exploration application with using your text and images.
    *   Build complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo’s query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

*   **🍱 Managed cloud**
    *   Low latency optimised deployment of Marqo.
    *   Scale inference at the click of a button.
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

## Get Started Quickly

1.  **Prerequisites:** Marqo requires Docker.  Install it from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB memory and 50GB storage.

2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```

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

    *   `mq` is the client that wraps the `marqo` API.
    *   `create_index()` creates a new index. Specify your model using `model="model-name"`.
    *   `add_documents()` takes a list of documents for indexing. `tensor_fields` specifies fields to be indexed as vector collections.

    See the example code in the original README for details on interpreting the results.

## Other Basic Operations

### Get document

Retrieve a document by ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

Note that by adding the document using ```add_documents``` again using the same ```_id``` will cause a document to be updated.

### Get index stats

Get information about an index.

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical search

Perform a keyword search.

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi modal and cross modal search

To power image and text search, Marqo allows users to plug and play with CLIP models from HuggingFace. **Note that if you do not configure multi modal search, image urls will be treated as strings.** To start indexing and searching with images, first create an index with a CLIP configuration, as below:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Images can then be added within documents as follows. You can use urls from the internet (for example S3) or from the disk of the machine:

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

You can then search the image field using text.

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

### Delete documents

Delete documents.

```python

results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])

```

### Delete index

Delete an index.

```python
results = mq.index("my-first-index").delete()
```

## Deployment Options

*   **Kubernetes:**  Deploy Marqo in production using our Kubernetes templates.
*   **Marqo Cloud:**  For a fully managed experience, sign up at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Access the comprehensive Marqo documentation [here](https://docs.marqo.ai/).

## Important Considerations

*   Do not run other applications on Marqo's Vespa cluster, as Marqo automatically manages and adapts its settings.

## Contribute

Marqo thrives on community contributions!  Please review the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests: `tox`
5.  If you update dependencies, delete the `.tox` directory and rerun `tox`.

## Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   Discuss and share ideas on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack).