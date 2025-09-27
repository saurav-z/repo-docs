<div align='center'>

![Chonkie Logo](https://github.com/chonkie-inc/chonkie/blob/main/assets/chonkie_logo_br_transparent_bg.png?raw=true)

# 🦛 Chonkie: The Ultra-Fast, Lightweight Text Chunking Library

**Tired of slow, bloated chunking libraries?** Chonkie is the perfect solution for efficiently processing your text data.

[Installation](#installation) •
[Usage](#basic-usage) •
[Pipeline](#the-chonkie-pipeline) •
[Chunkers](#chunkers) •
[Integrations](#integrations) •
[Benchmarks](#benchmarks)

Chonkie is a Python library designed for **lightning-fast and efficient text chunking**, making it ideal for Retrieval-Augmented Generation (RAG) and other NLP applications.

**Key Features:**

*   ✅ **Feature-Rich:** All the CHONKs you'll ever need.
*   ✨ **Easy to Use:** Simple installation, import, and chunking.
*   ⚡ **Fast:** Built for speed, get your chunks quickly.
*   🪶 **Lightweight:** Minimal dependencies for a small footprint.
*   🌏 **Wide Support:** Integrates with popular tokenizers, embedding models, and APIs.
*   💬 **Multilingual:** Out-of-the-box support for 56 languages.
*   ☁️ **Cloud-Ready:** Run locally or in the [Chonkie Cloud](https://cloud.chonkie.ai).

## Installation

Install Chonkie easily using pip:

```bash
pip install chonkie
```

For specific features, install optional dependencies:

```bash
pip install chonkie[all] # Installs all optional dependencies (not recommended for production)
```

Refer to our [docs](https://docs.chonkie.ai) for detailed installation instructions and to install only the dependencies you need.

## Basic Usage

Here's a quick example to get started with Chonkie:

```python
from chonkie import RecursiveChunker

chunker = RecursiveChunker()
chunks = chunker("Chonkie is the goodest boi! My favorite chunking hippo hehe.")

for chunk in chunks:
    print(f"Chunk: {chunk.text}")
    print(f"Tokens: {chunk.token_count}")
```

Explore more examples in the [docs](https://docs.chonkie.ai)!

## The Chonkie Pipeline

Chonkie utilizes a modular pipeline (`CHOMP`) to transform your text into high-quality chunks, offering flexibility and efficiency.

**Pipeline Stages:**

1.  📄 **Document:** Your input text data.
2.  👨‍🍳 **Chef:** Optional preprocessing (cleaning, normalization).
3.  🦛 **Chunker:** Core component for splitting text (e.g., RecursiveChunker).
4.  🏭 **Refinery:** Post-processing, such as merging and embedding.
5.  🤗 **Friends:** Export chunks and ingest them into your vector database.
    *   🐴 **Porters:** Export chunks to a file or a database.
    *   🤝 **Handshakes:** Ingest chunks into your preferred vector databases.

![🤖 CHOMP pipeline diagram](./assets/chomp-transparent-bg.png)

This modular design makes Chonkie powerful and easy to configure.

## Chunkers

Chonkie offers a variety of chunkers to suit your specific needs:

| Name               | Alias       | Description                                                                                                                |
| ------------------ | ----------- | -------------------------------------------------------------------------------------------------------------------------- |
| `TokenChunker`     | `token`     | Splits text into fixed-size token chunks.                                                                                  |
| `SentenceChunker`  | `sentence`  | Splits text into chunks based on sentences.                                                                                |
| `RecursiveChunker` | `recursive` | Splits text hierarchically using customizable rules to create semantically meaningful chunks.                              |
| `SemanticChunker`  | `semantic`  | Splits text into chunks based on semantic similarity. Inspired by the work of [Greg Kamradt](https://github.com/gkamradt). |
| `LateChunker`      | `late`      | Embeds text and then splits it to have better chunk embeddings.                                                            |
| `CodeChunker`      | `code`      | Splits code into structurally meaningful chunks.                                                                           |
| `NeuralChunker`    | `neural`    | Splits text using a neural model.                                                                                          |
| `SlumberChunker`   | `slumber`   | Splits text using an LLM to find semantically meaningful chunks. Also known as _"AgenticChunker"_.                         |

Learn more about each method in the [docs](https://docs.chonkie.ai).

## Integrations

Chonkie seamlessly integrates with 24+ tokenizers, embedding providers, LLMs, refineries, porters, vector databases, and utilities.

<details>
<summary><strong>Tokenizers</strong></summary>

Chonkie supports multiple tokenization methods:

| Name           | Description                                                    | Optional Install      |
| -------------- | -------------------------------------------------------------- | --------------------- |
| `character`    | Basic character-level tokenizer. **Default tokenizer.**        | `default`             |
| `word`         | Basic word-level tokenizer.                                    | `default`             |
| `tokenizers`   | Load any tokenizer from the Hugging Face `tokenizers` library. | `chonkie[tokenizers]` |
| `tiktoken`     | Use OpenAI's `tiktoken` library (e.g., for `gpt-4`).           | `chonkie[tiktoken]`   |
| `transformers` | Load tokenizers via `AutoTokenizer` from HF `transformers`.    | `chonkie[neural]`     |

Use a custom token counter for even more flexibility:

```python
def custom_token_counter(text: str) -> int:
    return len(text)

chunker = RecursiveChunker(tokenizer_or_token_counter=custom_token_counter)
```

</details>

<details>
<summary><strong>Embeddings</strong></summary>

Integrate with various embedding model providers:

| Provider / Alias        | Class                           | Description                            | Optional Install        |
| ----------------------- | ------------------------------- | -------------------------------------- | ----------------------- |
| `model2vec`             | `Model2VecEmbeddings`           | Use `Model2Vec` models.                | `chonkie[model2vec]`    |
| `sentence-transformers` | `SentenceTransformerEmbeddings` | Use any `sentence-transformers` model. | `chonkie[st]`           |
| `openai`                | `OpenAIEmbeddings`              | Use OpenAI's embedding API.            | `chonkie[openai]`       |
| `azure-openai`          | `AzureOpenAIEmbeddings`         | Use Azure OpenAI embedding service.    | `chonkie[azure-openai]` |
| `cohere`                | `CohereEmbeddings`              | Use Cohere's embedding API.            | `chonkie[cohere]`       |
| `gemini`                | `GeminiEmbeddings`              | Use Google's Gemini embedding API.     | `chonkie[gemini]`       |
| `jina`                  | `JinaEmbeddings`                | Use Jina AI's embedding API.           | `chonkie[jina]`         |
| `voyageai`              | `VoyageAIEmbeddings`            | Use Voyage AI's embedding API.         | `chonkie[voyageai]`     |

</details>

<details>
<summary><strong>LLMs (Genies)</strong></summary>

Use LLMs for advanced chunking:

| Genie Name     | Class              | Description                       | Optional Install        |
| -------------- | ------------------ | --------------------------------- | ----------------------- |
| `gemini`       | `GeminiGenie`      | Interact with Google Gemini APIs. | `chonkie[gemini]`       |
| `openai`       | `OpenAIGenie`      | Interact with OpenAI APIs.        | `chonkie[openai]`       |
| `azure-openai` | `AzureOpenAIGenie` | Interact with Azure OpenAI APIs.  | `chonkie[azure-openai]` |

</details>

<details>
<summary><strong>Refineries</strong></summary>

Enhance your chunks with:

| Refinery Name | Class                | Description                                   | Optional Install    |
| ------------- | -------------------- | --------------------------------------------- | ------------------- |
| `overlap`     | `OverlapRefinery`    | Merge overlapping chunks based on similarity. | `default`           |
| `embeddings`  | `EmbeddingsRefinery` | Add embeddings to chunks using any provider.  | `chonkie[semantic]` |

</details>

<details>
<summary><strong>Porters</strong></summary>

Save your chunks using:

| Porter Name | Class            | Description                            | Optional Install    |
| ----------- | ---------------- | -------------------------------------- | ------------------- |
| `json`      | `JSONPorter`     | Export chunks to a JSON file.          | `default`           |
| `datasets`  | `DatasetsPorter` | Export chunks to HuggingFace datasets. | `chonkie[datasets]` |

</details>

<details>
<summary><strong>Handshakes (Vector Databases)</strong></summary>

Ingest chunks into your favorite vector databases:

| Handshake Name | Class                  | Description                                  | Optional Install    |
| -------------- | ---------------------- | -------------------------------------------- | ------------------- |
| `chroma`       | `ChromaHandshake`      | Ingest chunks into ChromaDB.                 | `chonkie[chroma]`   |
| `qdrant`       | `QdrantHandshake`      | Ingest chunks into Qdrant.                   | `chonkie[qdrant]`   |
| `pgvector`     | `PgvectorHandshake`    | Ingest chunks into PostgreSQL with pgvector. | `chonkie[pgvector]` |
| `turbopuffer`  | `TurbopufferHandshake` | Ingest chunks into Turbopuffer.              | `chonkie[tpuf]`     |
| `pinecone`     | `PineconeHandshake`    | Ingest chunks into Pinecone.                 | `chonkie[pinecone]` |
| `weaviate`     | `WeaviateHandshake`    | Ingest chunks into Weaviate.                 | `chonkie[weaviate]` |
| `mongodb`      | `MongoDBHandshake`     | Ingest chunks into MongoDB.                  | `chonkie[mongodb]`  |

</details>

<details>
<summary><strong>Utilities</strong></summary>

Additional helpful utilities:

| Utility Name | Class        | Description                                    | Optional Install |
| ------------ | ------------ | ---------------------------------------------- | ---------------- |
| `hub`        | `Hubbie`     | Simple wrapper for HuggingFace Hub operations. | `chonkie[hub]`   |
| `viz`        | `Visualizer` | Rich console visualizations for chunks.        | `chonkie[viz]`   |

</details>

<details>
<summary><strong>Chefs & Fetchers</strong></summary>

Text preprocessing and data loading components:

| Component | Class         | Description                           | Optional Install |
| --------- | ------------- | ------------------------------------- | ---------------- |
| `chef`    | `TextChef`    | Text preprocessing and cleaning.      | `default`        |
| `fetcher` | `FileFetcher` | Load text from files and directories. | `default`        |

</details>

## Benchmarks

Chonkie excels in both speed and size:

**Size:**

*   **Default Install:** 15MB (vs 80-171MB for alternatives)
*   **With Semantic:** Still 10x lighter than the competition!

**Speed:**

*   **Token Chunking:** 33x faster than the slowest alternative
*   **Sentence Chunking:** Almost 2x faster than competitors
*   **Semantic Chunking:** Up to 2.5x faster than others

See detailed benchmarks in [BENCHMARKS.md](BENCHMARKS.md).

## Contributing

Contribute to Chonkie! See [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines. Every contribution improves Chonkie.

## Acknowledgements

Thanks to all users and contributors who help make Chonkie great! Special thanks to [Moto Moto](https://www.youtube.com/watch?v=I0zZC4wtqDQ&t=5s).

## Citation

If you use Chonkie, cite it as follows:

```bibtex
@software{chonkie2025,
  author = {Minhas, Bhavnick AND Nigam, Shreyash},
  title = {Chonkie: A no-nonsense fast, lightweight, and efficient text chunking library},
  year = {2025},
  publisher = {GitHub},
  howpublished = {\url{https://github.com/chonkie-inc/chonkie}},
}
```

</div>

**[Back to Top](https://github.com/chonkie-inc/chonkie)**