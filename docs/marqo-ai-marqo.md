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

**Marqo simplifies building cutting-edge search applications by offering an end-to-end vector search solution that handles vector generation, storage, and retrieval, all through a single API.** Explore the [Marqo repository](https://github.com/marqo-ai/marqo) to get started!

**Key Features:**

*   **🤖 State-of-the-art Embeddings:** Leverage the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more. Bring your own models or start with pre-configured options, with both CPU and GPU support.
*   **⚡ High Performance:** Experience blazing-fast search speeds with embeddings stored in-memory HNSW indexes and scale to hundreds of millions of documents with horizontal index sharding. Enjoy async and non-blocking data upload and search.
*   **🌌 Documents-in-Documents-out:**  Effortlessly build search, entity resolution, and data exploration applications using your text and images. Generate vectors, store data, and retrieve results out-of-the-box, supporting a wide range of data types.
*   **🍱 Managed Cloud:**  Benefit from low-latency optimized deployment, easy scaling of inference, high availability, 24/7 support, and access control with Marqo Cloud. Learn more [here](https://www.marqo.ai/cloud).
*   **🖼️ Multimodal Search:** Seamlessly integrate text and images for powerful search capabilities, using the power of CLIP models.
*   **🧮 Weighted Queries:** Construct advanced search queries with weighted terms and negations for precise and nuanced results.
*   **🔌  Easy Integration:** Seamlessly integrates with popular AI and data processing frameworks, including Haystack, Griptape, Langchain, and Hamilton.

## Core Features

Marqo offers a range of core features designed to make vector search accessible and powerful:
*   **State-of-the-art embeddings**
    *   Use the latest machine learning models from PyTorch, Huggingface, OpenAI and more. 
    *   Start with a pre-configured model or bring your own.
    *   CPU and GPU support.

*   **Performance**
    *   Embeddings stored in in-memory HNSW indexes, achieving cutting edge search speeds.
    *   Scale to hundred-million document indexes with horizontal index sharding.
    *   Async and non-blocking data upload and search.

*   **Documents-in-documents-out**
    *   Vector generation, storage, and retrieval are provided out of the box.
    *   Build search, entity resolution, and data exploration application with using your text and images.
    *   Build complex semantic queries by combining weighted search terms.
    *   Filter search results using Marqo’s query DSL.
    *   Store unstructured data and semi-structured metadata together in documents, using a range of supported datatypes like bools, ints and keywords.

*   **Managed cloud**
    *   Low latency optimised deployment of Marqo.
    *   Scale inference at the click of a button.
    *   High availability.
    *   24/7 support.
    *   Access control.
    *   Learn more [here](https://www.marqo.ai/cloud).

## Quick Start

Get up and running with Marqo in minutes!

1.  **Install Docker:**  If you don't have Docker installed, go to the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker has at least 8GB memory and 50GB storage.

2.  **Run Marqo using Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -it -p 8882:8882 marqoai/marqo:latest
    ```
    *Note:  If the container is killed, increase Docker memory (8GB recommended) in Docker settings.*

3.  **Install the Marqo client:**

    ```bash
    pip install marqo
    ```

4.  **Start Indexing and Searching!**

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

    *   `mq` is the Marqo API client.
    *   `create_index()` creates a new index. Specify a model, e.g., `hf/e5-base-v2`.  Experiment to find the best model for your use case.
    *   `add_documents()` adds documents for indexing. `tensor_fields` specifies fields for vector indexing.
    *   You can use the special `_id` field to set a document ID.  Otherwise, Marqo will generate one.

    View results:

    ```python
    import pprint
    pprint.pprint(results)
    ```

    *   Each `hit` is a document matching the query, ordered by relevance.
    *   `_highlights` show the matching parts of the document.

##  Other Basic Operations

*   **Get document:** Retrieve by ID:
    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```

*   **Get index stats:**
    ```python
    results = mq.index("my-first-index").get_stats()
    ```

*   **Lexical search:** Keyword search:
    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```

*   **Multimodal and Cross-Modal Search**:  Use CLIP models for image and text search:
    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)

    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])

    results = mq.index("my-multimodal-index").search('animal')
    ```

*   **Image search**: Search using an image URL:
    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```

*   **Weighted Queries:**
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

*   **Creating and searching indexes with multimodal combination fields**

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

*   **Delete documents:**
    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```

*   **Delete index:**
    ```python
    results = mq.index("my-first-index").delete()
    ```

## Production Deployment

Marqo offers flexible deployment options:

*   **Kubernetes:**  We provide Kubernetes templates for deploying Marqo on your preferred cloud provider.  Find the repo at: [https://github.com/marqo-ai/marqo-on-kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes)

*   **Marqo Cloud:** For a fully managed experience, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Detailed documentation is available at: [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Considerations

*   Do not run other applications on Marqo's Vespa cluster as it automatically adjusts its settings.

## Contributing

Marqo is a community project, and contributions are welcome!  See [CONTRIBUTING.md](./CONTRIBUTING.md) to get started.

## Development Setup

1.  Create a virtual environment:  `python -m venv ./venv`
2.  Activate the environment:  `source ./venv/bin/activate`
3.  Install requirements:  `pip install -r requirements.txt`
4.  Run tests:  `tox`
5.  If you update dependencies, delete `.tox` and rerun `tox`.

## Merge Instructions

1.  Run the full test suite: `tox`
2.  Create a pull request with an attached GitHub issue.

## Support

*   **Discourse:** Ask questions and share on our [Discourse forum](https://community.marqo.ai).
*   **Slack:** Join our [Slack community](https://bit.ly/marqo-community-slack).