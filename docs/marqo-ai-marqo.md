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

**Marqo is a cutting-edge vector search engine that simplifies building semantic search applications for both text and images, handling vector generation, storage, and retrieval with a single API.** ([See the GitHub repository](https://github.com/marqo-ai/marqo))

**Key Features:**

*   **🤖 State-of-the-art Embeddings:** Utilizes the latest machine learning models from PyTorch, Hugging Face, OpenAI, and more, with both pre-configured and custom model options.
*   **⚡ High Performance:** Leverages in-memory HNSW indexes for rapid search speeds and scales to handle indexes with hundreds of millions of documents.  Offers async and non-blocking data upload and search.
*   **🌌 Documents-in-Documents-out:** Simplifies your workflow by handling vector generation, storage, and retrieval out of the box for both text and images.
*   **🍱 Managed Cloud:** Provides a fully managed cloud service with low-latency deployments, scalable inference, high availability, 24/7 support, and access control; learn more [here](https://www.marqo.ai/cloud).
*   **💙 Integrations:** Seamlessly integrates with popular AI and data processing frameworks like Haystack, Griptape, Langchain, and Hamilton.
*   **🖼️ Multimodal Search:** Perform image and text search.

### Learn More and Get Started

*   **[Quick Start](#getting-started):** Build your first application with Marqo in under 5 minutes.
*   **[Marqo for Image Data](https://www.marqo.ai/blog/context-is-all-you-need-multimodal-vector-search-with-personalization):** Learn how to build advanced image search with Marqo.
*   **[Marqo for Text](https://www.marqo.ai/blog/how-i-used-marqo-to-create-a-multilingual-legal-databse-in-5-key-lines-of-code):** Learn how to build a multilingual database in Marqo.
*   **[Integrating Marqo with GPT](https://www.marqo.ai/blog/from-iron-manual-to-ironman-augmenting-gpt-with-marqo-for-fast-editable-memory-to-enable-context-aware-question-answering):** Learn how to make GPT a subject matter expert by using Marqo as a knowledge base.
*   **[Marqo for Creative AI](https://www.marqo.ai/blog/combining-stable-diffusion-with-semantic-search-generating-and-categorising-100k-hot-dogs):** Learn how to combine stable diffusion with semantic search to generate and categorise 100k images of hotdogs.
*   **[Marqo and Speech Data](https://www.marqo.ai/blog/speech-processing):** Learn how to add diarisation and transcription to preprocess audio for Q&A with Marqo and ChatGPT.
*   **[Marqo for content moderation](https://www.marqo.ai/blog/refining-image-quality-and-eliminating-nsfw-content-with-marqo):** Learn how to build advanced image search with Marqo to find and remove content.
*   **[Getting started with Marqo Cloud](https://github.com/marqo-ai/getting_started_marqo_cloud):** Go through how to get set up and running with Marqo Cloud starting from your first time login through to building your first application with Marqo
*   **[Marqo for e-commerce](https://github.com/marqo-ai/getting_started_marqo_cloud/blob/main/e-commerce-demo/README.md):** This project is a web application with frontend and backend using Python, Flask, ReactJS, and Typescript. The frontend is a ReactJS application that makes requests to the backend which is a Flask application. The backend makes requests to your Marqo cloud API.
*   **[Marqo chatbot](https://github.com/marqo-ai/getting_started_marqo_cloud/tree/main/chatbot-demo):** In this guide we will build a chat bot application using Marqo and OpenAI's ChatGPT API. We will start with an existing code base and then walk through how to customise the behaviour.

## Getting Started

Follow these steps to quickly set up and use Marqo:

1.  **Install Docker:**  Download and install Docker from the [Docker Official website](https://docs.docker.com/get-docker/). Ensure Docker is configured with at least 8GB of memory (16GB recommended) and 50GB of storage.
2.  **Run Marqo with Docker:**

    ```bash
    docker rm -f marqo
    docker pull marqoai/marqo:latest
    docker run --name marqo -p 8882:8882 marqoai/marqo:latest
    ```

    *   *Note: If the `marqo` container keeps getting killed, increase the memory limit for Docker in your settings, to at least 6GB (8GB recommended).*
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
    *   `create_index()` creates a new index; the `model` parameter specifies the embedding model (e.g., `"hf/e5-base-v2"`).  Experimentation with different models is often required to achieve the best retrieval for your specific use case. See [here](https://docs.marqo.ai/1.0.0/Models-Reference/dense_retrieval/) for the full list of models.
    *   `add_documents()` adds documents for indexing, using Python dicts. `tensor_fields` specifies fields to be indexed as vector collections.
    *   Use the special `_id` field to set a document's ID; otherwise, Marqo generates one.

    ```python
    import pprint
    pprint.pprint(results)
    ```

    *   `hits` are the documents that match the search query, ordered by relevance.
    *   `limit` sets the maximum number of hits returned.
    *   `_highlights` shows the portion of the document that best matched the query.

## Other Basic Operations

*   **Get Document:** Retrieve a document by ID.

    ```python
    result = mq.index("my-first-index").get_document(document_id="article_591")
    ```

    *   *Note that by adding the document using ```add_documents``` again using the same ```_id``` will cause a document to be updated.*

*   **Get Index Stats:** Get information about an index.

    ```python
    results = mq.index("my-first-index").get_stats()
    ```

*   **Lexical Search:** Perform a keyword search.

    ```python
    result = mq.index("my-first-index").search('marco polo', search_method=marqo.SearchMethods.LEXICAL)
    ```

*   **Multimodal and Cross-Modal Search:** Integrate text and image search, using CLIP models.  *Note that if you do not configure multimodal search, image urls will be treated as strings.* Create an index with the appropriate settings:

    ```python
    settings = {
        "treat_urls_and_pointers_as_images":True,   # allows us to find an image file and index it
        "model":"ViT-L/14"
    }
    response = mq.create_index("my-multimodal-index", **settings)
    ```

    Add images within documents:

    ```python
    response = mq.index("my-multimodal-index").add_documents([{
        "My_Image": "https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_realistic.png",
        "Description": "The hippopotamus, also called the common hippopotamus or river hippopotamus, is a large semiaquatic mammal native to sub-Saharan Africa",
        "_id": "hippo-facts"
    }], tensor_fields=["My_Image"])
    ```

    Search images with text:

    ```python
    results = mq.index("my-multimodal-index").search('animal')
    ```

*   **Searching with Images:** Search using an image link directly.

    ```python
    results = mq.index("my-multimodal-index").search('https://raw.githubusercontent.com/marqo-ai/marqo-api-tests/mainline/assets/ai_hippo_statue.png')
    ```

*   **Searching with Weighted Queries:** Use dictionaries to define queries with weighted terms for more complex and nuanced searches, including negation.

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

*   **Multimodal Combination Fields:** Create indexes with multimodal combination fields to combine text and images into one field for scoring and single vector representation.

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

*   **Delete Documents:** Delete documents by ID.

    ```python
    results = mq.index("my-first-index").delete_documents(ids=["article_591", "article_602"])
    ```

*   **Delete Index:** Delete an index.

    ```python
    results = mq.index("my-first-index").delete()
    ```

## Running Marqo in Production

*   **Kubernetes:** Marqo offers Kubernetes templates for deployment on cloud providers.  See the [Marqo on Kubernetes](https://github.com/marqo-ai/marqo-on-kubernetes) repository.
*   **Marqo Cloud:** For a fully managed service, sign up for Marqo Cloud at [https://cloud.marqo.ai](https://cloud.marqo.ai).

## Documentation

Find comprehensive documentation at [https://docs.marqo.ai/](https://docs.marqo.ai/).

## Important Note

Do not run other applications on Marqo's Vespa cluster, as Marqo automatically adjusts the cluster's settings.

## Contributing

Marqo is a community project; we welcome your contributions!  See the [CONTRIBUTING.md](./CONTRIBUTING.md) file to get started.

## Development Setup

1.  Create a virtual environment: `python -m venv ./venv`
2.  Activate the environment: `source ./venv/bin/activate`
3.  Install requirements: `pip install -r requirements.txt`
4.  Run tests: Run `tox` in the project directory.
5.  If you update dependencies, delete the `.tox` directory and rerun tests.

## Merge Instructions

1.  Run the full test suite: `tox`.
2.  Create a pull request with a linked GitHub issue.

## Support

*   Engage with the community on our [Discourse forum](https://community.marqo.ai).
*   Join our [Slack community](https://bit.ly/marqo-community-slack) for discussions and support.