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

Marqo is an open-source vector search engine that simplifies and accelerates your ability to build intelligent search applications for both text and images; visit the [Marqo GitHub](https://github.com/marqo-ai/marqo) to get started today!

**Key Features:**

*   **⚡️ High Performance:** Utilize in-memory HNSW indexes for cutting-edge search speeds, and scale to hundreds of millions of documents with horizontal index sharding.
*   **🤖 State-of-the-Art Embeddings:** Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more, with options to bring your own models.
*   **🌌 Documents-In-Documents-Out:** Simplify vector generation, storage, and retrieval for building search, entity resolution, and data exploration applications.
*   **🍱 Managed Cloud Option:** Deploy Marqo with low latency optimized deployment, and scale inference at the click of a button with high availability and 24/7 support.
*   **🖼️ Multimodal Search:** Easily index and search images using text queries with built-in support for CLIP models, making it simple to build image and text search applications.
*   **📦 Integrations:**  Seamlessly integrate with popular AI and data processing frameworks, including Haystack, Griptape, Langchain, and Hamilton.

## Core Features

**🤖 State of the art embeddings**
- Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more. 
- Start with a pre-configured model or bring your own.
- CPU and GPU support.

**⚡ Performance**
- Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
- Scale to hundred-million document indexes with horizontal index sharding.
- Async and non-blocking data upload and search.

**🌌 Documents-in-documents-out**
- Vector generation, storage, and retrieval are provided out of the box.
- Build search, entity resolution, and data exploration application with using your text and images.
- Build complex semantic queries by combining weighted search terms.
- Filter search results using Marqo’s query DSL.
- Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

**🍱 Managed cloud**
- Low latency optimised deployment of Marqo.
- Scale inference at the click of a button.
- High availability.
- 24/7 support.
- Access control.
- Learn more [here](https://www.marqo.ai/cloud).

## Integrations

Marqo is integrated into popular AI and data processing frameworks, with more on the way.

**💙 [Haystack](https://github.com/deepset-ai/haystack)**

Haystack is an open-source framework for building applications that make use of NLP technology such as LLMs, embedding models and more. This [integration](https://haystack.deepset.ai/integrations/marqo-document-store) allows you to use Marqo as your Document Store for Haystack pipelines such as retrieval-augmentation, question answering, document search and more.

**🛹 [Griptape](https://github.com/griptape-ai/griptape)**

Griptape enables safe and reliable deployment of LLM-based agents for enterprise applications, the MarqoVectorStoreDriver gives these agents access to scalable search with your own data. This integration lets you leverage open source or custom fine-tuned models through Marqo to deliver relevant results to your LLMs.

**🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**

This integration lets you leverage open source or custom fine tuned models through Marqo for LangChain applications with a vector search component. The Marqo vector store implementation can plug into existing chains such as the Retrieval QA and Conversational Retrieval QA.

**⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

This integration lets you leverage open source or custom fine tuned models through Marqo for Hamilton LLM applications. 

## Learn More about Marqo
                                                                                                                                                       
| | |
| --------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)| Build your first application with Marqo in under 5 minutes. |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization) | Building advanced image search with Marqo. |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code) | Building a multilingual database in Marqo. |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Making GPT a subject matter expert by using Marqo as a knowledge base. |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs. |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing) | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT. |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo) | Building advanced image search with Marqo to find and remove content. |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud) | Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo|
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md) | This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.|
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo) | In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.|
| 🦾 [Features](#-Core-Features) | Marqo's core features. |

## Getting Started

1.  **Docker Installation:** Marqo requires Docker. Install Docker from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB of memory and 50GB of storage allocated.

2.  **Run Marqo with Docker:**

```bash
docker rm -f marqo
docker pull marqoai/marqo:latest
docker run --name marqo -p 8882:8882 marqoai/marqo:latest
```

    *Note:* If your `marqo` container is frequently killed, increase Docker's memory allocation to at least 6GB (8GB recommended) in your Docker settings.

3.  **Install Marqo Client:**

```bash
pip install marqo
```

4.  **Start Indexing and Searching:** Example code snippet:

```python
import marqo

mq = marqo.Client(url='http://localhost:8882')

mq.create_index("my-first-index", model="hf/e5-base-v2") # Specify model here

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

    *   `mq`: The client that wraps the `marqo` API.
    *   `create_index()`: Creates a new index with optional model specification, e.g., `mq.create_index("my-first-index", model="hf/e5-base-v2")`.
    *   `add_documents()`: Adds documents (represented as Python dicts) for indexing, with `tensor_fields` indicating the fields to index as vector collections.
    *   Documents can have a custom ID with the `_id` field. Otherwise, Marqo generates one.

    Let's examine the results:

```python
# print out the results:
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

    *   Each `hit` corresponds to a document that matches the search query.
    *   Results are ordered from most to least matching.
    *   `limit` is the maximum number of hits returned (can be set during search).
    *   Each hit has a `_highlights` field, showing the part of the document that best matched the query.

## Other Basic Operations

### Get Document

Retrieve a document by ID:

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

*Note:* Re-adding a document with the same `_id` updates the existing document.

### Get Index Stats

Retrieve information about an index:

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical Search

Perform a keyword search:

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multimodal and Cross-Modal Search

Marqo enables image and text search by using CLIP models from Hugging Face.  If you don't configure multimodal search, image URLs are treated as strings.

To enable image indexing and searching, create an index with a CLIP configuration:

```python
settings = {
    "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it 
    "model":"ViT-L/14"
}
response = mq.create_index("my-multimodal-index", **settings)
```

Add images within documents using URLs (e.g., S3) or local file paths:

```python
response = mq.index("my-multimodal-index").add_documents([{
    "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
    "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
    "_id": "hippo-facts"
}], tensor_fields=["My_Image"])
```

Then, search the image field using text:

```python
results = mq.index("my-multimodal-index").search('animal')
```

### Searching with an Image

Search using an image URL:

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching with Weighted Queries

Use dictionaries to define queries with weights for more complex searches:

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

Marqo supports multimodal combination fields for combining text and images, allowing for a single vector representation and weighted scoring of combined fields.

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

Delete documents using their IDs:

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete Index

Delete an index:

```python
results = mq.index("my-first-index").delete()
```

## Production Deployment

Marqo offers Kubernetes templates for deployment on your preferred cloud provider, supporting clusters with replicas, multiple storage shards, and inference nodes, deploy the solution found here: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).  For a fully managed cloud service, sign up for Marqo Cloud: [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Comprehensive Marqo documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Warning

Do not run other applications on Marqo's Vespa cluster, as Marqo automatically manages and adapts the settings on the cluster.

## Contributing

Marqo is a community-driven project.  We welcome your contributions!  Please review [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`

2.  Activate the environment: `source ./venv/bin/activate`

3.  Install requirements: `pip install -r requirements.txt`

4.  Run tests with `tox`.

5.  If you update dependencies, delete the `.tox` directory and rerun.

## Merge Instructions

1.  Run the full test suite (using `tox`).

2.  Create a pull request with an attached GitHub issue.

## Support

*   Discuss and share your projects on the [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack) for collaboration.