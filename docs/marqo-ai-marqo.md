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

**Unlock powerful semantic search capabilities with Marqo, the end-to-end vector search engine that simplifies embedding generation, storage, and retrieval through a single API.**  [Explore the original repo](https://github.com/marqo-ai/marqo).

**Key Features:**

*   **🤖 State-of-the-Art Embeddings:** Utilize the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more.  Easily integrate pre-configured or custom models with CPU and GPU support.
*   **⚡ High Performance:** Experience cutting-edge search speeds with embeddings stored in in-memory HNSW indexes.  Scale effortlessly with horizontal index sharding for indexes containing hundreds of millions of documents. Benefit from async and non-blocking data upload and search.
*   **🌌 Documents-in, Documents-out:** Simplify your workflow with out-of-the-box vector generation, storage, and retrieval. Build search, entity resolution, and data exploration applications with text and image data.  Create complex semantic queries by combining weighted search terms and filter results using Marqo’s query DSL. Store unstructured and semi-structured metadata using a variety of supported datatypes such as booleans, integers, and keywords.
*   **🍱 Managed Cloud (Optional):** Deploy Marqo with low latency optimization and enjoy scalable inference, high availability, and 24/7 support.  Learn more about Marqo Cloud [here](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Search:** Easily search both text and images using Marqo's built-in support for CLIP models, and create multimodal combination fields to combine text and images into one field.

## Quick Start

Get up and running with Marqo in minutes:

1.  **Install Docker:** Follow the instructions on the [Docker Official website](https://docs.docker.com/get-docker/) to install Docker. Ensure Docker has at least 8GB of memory and 50GB of storage.  In Docker Desktop, you can do this by clicking the settings icon, then resources, and selecting 8GB memory.
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

## Integrations

Marqo integrates seamlessly with popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## Learn More

|                                                                                       |                                                                                                                                                         |
| ------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#quick-start)                                                       | Build your first application with Marqo in under 5 minutes.                                                                                            |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo.                                                                                                  |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo.                                                                                                   |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base.                                                                               |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.                                                        |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                                                                        |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content.                                                                        |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo. |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.|
| 🦾 [Core Features](#-core-features) | Marqo's core features. |

## Other Basic Operations

### Get Document

Retrieve a document by ID.

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

Note that adding a document again using `add_documents` with the same `_id` will cause a document to be updated.

### Get Index Stats

Get information about an index.

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

Perform a keyword search.

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multimodal and Cross-Modal Search

Marqo allows users to use CLIP models from HuggingFace for image and text search.

```python
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Images can then be added within documents as follows:

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

Search using an image by providing the image link.

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching Using Weights in Queries

Queries can be provided as dictionaries with weights for more advanced queries.

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

Marqo lets you have indexes with multimodal combination fields. Multimodal combination fields can combine text and images into one field. This allows scoring of documents across the combined text and image fields together. It also allows for a single vector representation instead of needing many which saves on storage. The relative weighting of each component can be set per document.

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

Delete documents by ID.

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

Delete an index.

```python
results = mq.index("my-first-index").delete()
```

## Production Deployment

Marqo offers Kubernetes templates for deployment.  Find the repo here: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).  For a fully managed cloud service, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Important Note

Do not run other applications on Marqo's Vespa cluster, as Marqo manages and adjusts the settings automatically.

## Contributing

Marqo thrives on community contributions! Review the [CONTRIBUTING.md](./CONTRIBUTING.md) guide to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests:  `tox` (after `cd` into the directory).
5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with an attached GitHub issue.

## Support

*   Discuss ideas and ask questions in the [Discourse forum](https://community.marqo.ai).
*   Join the [Slack community](https://bit.ly/marqo-community-slack) for community interaction.