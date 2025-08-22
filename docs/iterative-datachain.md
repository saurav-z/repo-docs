# DataChain: Unleash the Power of Your Unstructured Data

**DataChain is a Python-based AI data warehouse designed to transform, analyze, and version unstructured data at scale.**  [Explore the original repository](https://github.com/iterative/datachain).

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Versatile Data Handling:**
    *   Supports a wide range of unstructured data types including images, videos, text, PDFs, JSONs, CSVs, and Parquet files.
    *   Seamlessly integrates with cloud storage services like S3, GCP, and Azure, as well as local file systems.
    *   Unites files and associated metadata into persistent, versioned, and columnar datasets.
*   **Python-First Approach:**
    *   Empowers you to work with Python objects and their fields, including float scores, strings, and even matrixes.
    *   Provides built-in parallelization and memory-efficient computing, enabling high-scale data processing with no SQL or Spark expertise required.
*   **Robust Data Enrichment and Processing:**
    *   Generate metadata effortlessly using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata, and execute advanced searches with vector embeddings.
    *   Offers high-performance vectorized operations on Python objects, such as sum, count, and average.
    *   Easily integrates with PyTorch and TensorFlow, or allows for seamless export back into storage.
*   **Efficient Processing Workflows:**
    *   **Delta Processing:** Processes only new or changed files or records.
    *   **Retry Processing:** Automatically reprocesses records with errors.
    *   **Combined Approach:** Process new data and fix errors in a single pipeline.

## Use Cases

1.  **ETL (Extract, Transform, Load):** A Pythonic framework for defining and executing transformations on unstructured data.
2.  **Analytics:** Provides a table-like API for in-depth analytics on data objects at scale.
3.  **Versioning:** Efficiently manage and version data without data duplication, ideal for large datasets of images, videos, and documents.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

For in-depth information, examples, and a quick start guide, please visit our [documentation](https://docs.datachain.ai/).

## Examples

### Download Subset of Files Based on Metadata

This example demonstrates how to download a specific subset of files from cloud storage based on metadata.

```python
import datachain as dc

meta = dc.read_json("gs://datachain-demo/dogs-and-cats/*json", column="meta", anon=True)
images = dc.read_storage("gs://datachain-demo/dogs-and-cats/*jpg", anon=True)

images_id = images.map(id=lambda file: file.path.split('.')[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")

likely_cats = annotated.filter((dc.Column("meta.inference.confidence") > 0.93) \
                               & (dc.Column("meta.inference.class_") == "cat"))
likely_cats.to_storage("high-confidence-cats/", signal="file")
```

### Incremental Processing with Error Handling

This example shows how to use both delta and retry processing for efficient handling of large datasets.

```python
import datachain as dc
from datachain import C, File

def process_file(file: File):
    """Process a file, which may occasionally fail."""
    try:
        # Your processing logic here
        content = file.read_text()
        result = analyze_content(content)
        return {
            "content": content,
            "result": result,
            "error": None  # No error
        }
    except Exception as e:
        # Return an error that will trigger reprocessing next time
        return {
            "content": None,
            "result": None,
            "error": str(e)  # Error field will trigger retry
        }

# Process files efficiently with delta and retry
chain = (
    dc.read_storage(
        "data/",
        update=True,
        delta=True,              # Process only new/changed files
        delta_on="file.path",    # Identify files by path
        retry_on="error"         # Field that indicates errors
    )
    .map(processed_result=process_file)
    .mutate(
        content=C("processed_result.content"),
        result=C("processed_result.result"),
        error=C("processed_result.error")
    )
    .save(name="processed_data")
)
```

### LLM based text-file evaluation

This example evaluates chatbot conversations using LLM based evaluation.

```python
import os
from mistralai import Mistral
import datachain as dc

PROMPT = "Was this dialog successful? Answer in a single word: Success or Failure."

def eval_dialogue(file: dc.File) -> bool:
     client = Mistral(api_key = os.environ["MISTRAL_API_KEY"])
     response = client.chat.complete(
         model="open-mixtral-8x22b",
         messages=[{"role": "system", "content": PROMPT},
                   {"role": "user", "content": file.read()}])
     result = response.choices[0].message.content
     return result.lower().startswith("success")

chain = (
   dc.read_storage("gs://datachain-demo/chatbot-KiT/", column="file", anon=True)
   .settings(parallel=4, cache=True)
   .map(is_success=eval_dialogue)
   .save("mistral_files")
)

successful_chain = chain.filter(dc.Column("is_success") == True)
successful_chain.to_storage("./output_mistral")

print(f"{successful_chain.count()} files were exported")
```

## Contributing

We welcome contributions!  Please see our [Contributor Guide](https://docs.datachain.ai/contributing) to get started.

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution designed for teams, providing:

*   Centralized dataset registry.
*   Data lineage.
*   UI for Multimodal Data.
*   Scalable Compute.
*   Access control.