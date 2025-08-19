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

# Marqo: The Open-Source, End-to-End Vector Search Engine for Text & Images

**Marqo simplifies vector search by providing vector generation, storage, and retrieval through a single, easy-to-use API.**  Built for developers, Marqo lets you effortlessly build vector search into your applications.  Explore the [Marqo GitHub Repository](https://github.com/marqo-ai/marqo) to get started!

## 🚀 Key Features

*   **🤖 State-of-the-Art Embeddings:** Leverage the latest machine learning models from Hugging Face, OpenAI, and more, with both CPU and GPU support.
*   **⚡ High Performance:** Experience cutting-edge search speeds with in-memory HNSW indexes and scale to millions of documents with horizontal sharding. Benefit from async and non-blocking data upload and search.
*   **🌌 Documents-In-Documents-Out:**  Simplify your workflow with out-of-the-box vector generation, storage, and retrieval for both text and images.
*   **🍱 Managed Cloud:**  Benefit from a low-latency, scalable, and highly available deployment option with 24/7 support and access control.  Learn more about [Marqo Cloud](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Search:** Seamlessly index and search both text and images, including support for image URLs and on-disk images.
*   **⚖️ Flexible Querying:**  Build complex semantic queries using weighted search terms, Marqo's query DSL, and document filtering with boolean, integer, keyword datatypes.
*   **🔌 Integrations:**  Easily integrate with popular tools like Haystack, Griptape, Langchain, and Hamilton for enhanced functionality.

## 🔌 Integrations

Marqo integrates with several popular AI and data processing frameworks:

*   **💙 [Haystack](https://github.com/deepset-ai/haystack)**
*   **🛹 [Griptape](https://github.com/griptape-ai/griptape)**
*   **🦜🔗 [Langchain](https://github.com/langchain-ai/langchain)**
*   **⋙ [Hamilton](https://github.com/DAGWorks-Inc/hamilton/)**

## 📚 Learn More

Dive deeper into Marqo's capabilities with these resources:

| Resource                                                                                                                                       | Description                                                                                             |
| ---------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------- |
| 📗 [Quick start](#Getting-started)                                                                                                           | Build your first application with Marqo in under 5 minutes.                                             |
| 🖼 [Marqo for image data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization)                     | Build advanced image search with Marqo.                                                               |
| 📚 [Marqo for text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code)                 | Building a multilingual database in Marqo.                                                            |
| 🔮 [Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering) | Make GPT a subject matter expert by using Marqo as a knowledge base.                                 |
| 🎨 [ Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs) | Combining stable diffusion with semantic search to generate and categorise 100k images of hotdogs.        |
| 🔊 [Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing)                                                                        | Add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.                   |
| 🚫 [Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo)                | Building advanced image search with Marqo to find and remove content.                                 |
| ☁️ [Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud)                                                 | Go through how to get set up and running with Marqo Cloud starting from your first time login.       |
| 👗 [Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md)                            | A web application with frontend and backend using Python, Flask, ReactJS, and Typescript.                  |
| 🤖 [Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo)                                               | Building a chat bot application using Marqo and OpenAI's ChatGPT API.                                 |
| 🦾 [Features](#-Core-Features)                                                                                                                | Marqo's core features.                                                                                   |

## 🚀 Getting Started

Follow these steps to quickly set up and use Marqo:

1.  **Install Docker:** Ensure Docker is installed on your system (check the [Docker Official website](https://docs.docker.com/get-docker/)).  Allocate at least 8GB of memory and 50GB of storage in Docker settings.

2.  **Run Marqo with Docker:**
    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```
    *Note: If your `marqo` container keeps getting killed, increase Docker's memory limit to at least 6GB (8GB recommended).*

3.  **Install the Marqo Client:**
    ```bash
    pip install marqo
    ```

4.  **Start Indexing and Searching!**

    ```python
    import marqo

    mq = marqo.Client(url='http://localhost:8882')

    mq.create_index("my-first-index", model="hf/e5-base-v2")  # or specify a model

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
    pprint.pprint(results)
    ```

## 🧑‍💻 Other Basic Operations

### Get document

```python
result = mq.index("my-first-index").get_document(document_id="article_591")
```

### Get index stats

```python
results = mq.index("my-first-index").get_stats()
```

### Lexical search

```python
result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
```

### Multi modal and cross modal search

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

```python
results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
```

### Searching using weights in queries

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

### Delete documents

```python
results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
```

### Delete index

```python
results = mq.index("my-first-index").delete()
```

## 🏢 Production Deployment

Marqo offers flexible deployment options:

*   **Kubernetes:** Deploy Marqo on your preferred cloud provider using our Kubernetes templates: [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes).
*   **Marqo Cloud:** For a fully managed experience, sign up for [Marqo Cloud](https://cloud.marqo.ai).

## ⚠️ Important Notes

*   Avoid running other applications on Marqo's Vespa cluster, as Marqo automatically manages the cluster settings.

## 🤝 Contributing

Marqo thrives on community contributions! Review the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## 🧑‍💻 Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install dependencies: `pip install -r requirements.txt`
4.  Run tests: Navigate to the project directory and run `tox`.
5.  If you update dependencies, delete the `.tox` directory and rerun `tox`.

## 📝 Merge Instructions

1.  Run the full test suite using `tox`.
2.  Create a pull request with an attached GitHub issue.

## 💬 Support

*   **Discourse:** Ask questions and share your creations on our [Discourse forum](https://community.marqo.ai).
*   **Slack:** Join our [Slack community](https://bit.ly/marqo-community-slack) to connect with other community members.