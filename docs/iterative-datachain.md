# DataChain: Your Pythonic AI-Data Warehouse for Unstructured Data

**DataChain empowers you to transform, analyze, and version unstructured data like images, audio, videos, text, and PDFs with ease.**

[Go to the DataChain GitHub Repository](https://github.com/iterative/datachain)

## Key Features

*   **Effortless Versioning for Unstructured Data:**
    *   Version your data without the need to move or copy it.
    *   Works seamlessly with data stored on S3, GCP, Azure, and local file systems.
    *   Full support for various data types: images, videos, text, PDFs, JSONs, CSVs, and more.
    *   Unite files and metadata into versioned columnar datasets.
*   **Python-Native Functionality:**
    *   Operate directly on Python objects and their fields (float scores, strings, matrices, LLM response objects).
    *   Execute Python code on massive, terabyte-scale datasets with built-in parallelization and memory-efficient computing. No SQL or Spark required.
*   **Data Enrichment and Processing Capabilities:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata. Includes vector embeddings for searching.
    *   Benefit from high-performance vectorized operations on Python objects: sum, count, average, etc.
    *   Seamlessly integrate with PyTorch and TensorFlow, or export data back to storage.

## Use Cases

*   **ETL (Extract, Transform, Load):** A Pythonic framework for creating and running unstructured data transformations and enrichments, including applying models like LLMs.
*   **Analytics:** The DataChain dataset combines all the information about data objects in one place, and offers a DataFrame-like API and vectorized engine for large-scale analytics.
*   **Versioning:** DataChain doesn't require storing, moving, or copying data (unlike DVC), making it perfect for use cases such as a bucket with millions of images, videos, audio, and PDFs.
*   **Incremental Processing:** DataChain's delta and retry features allow for efficient processing workflows.
    *   **Delta Processing:** Process only new or changed files/records
    *   **Retry Processing:** Automatically reprocess records with errors or missing results
    *   **Combined Approach:** Process new data and fix errors in a single pipeline

## Getting Started

For detailed instructions and examples, visit the [DataChain Documentation](https://docs.datachain.ai/).

**Install DataChain:**

```bash
pip install datachain
```

## Community and Support

*   [Documentation](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## Contributing

We welcome contributions! Please refer to the [Contributor Guide](https://docs.datachain.ai/contributing) for more information.

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution for teams, which includes:

*   Centralized dataset registry
*   Data Lineage
*   UI for Multimodal Data
*   Scalable Compute
*   Access control