# DataChain: The Python-Powered AI-Data Warehouse for Unstructured Data

**DataChain** is a Python-based framework designed to transform, analyze, and version unstructured data like images, audio, videos, text, and PDFs with ease.  [Visit the original repository on GitHub](https://github.com/iterative/datachain) to learn more.

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Effortless Data Versioning:**
    *   Version unstructured data efficiently without data duplication by leveraging references to your existing cloud storage (S3, GCP, Azure) or local file systems.
    *   Supports a wide range of data types: images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unite files and metadata for persistent, versioned, and columnar datasets.

*   **Python-Friendly Data Manipulation:**
    *   Interact with Python objects and their fields directly: float scores, strings, matrices, and even LLM response objects.
    *   Run Python code at scale with built-in parallelization and memory-efficient computing, eliminating the need for SQL or Spark.

*   **Powerful Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Easily filter, join, and group datasets based on metadata, and perform vector-based searches.
    *   Utilize high-performance vectorized operations on Python objects for tasks like summing, counting, and averaging.
    *   Seamlessly integrate with Pytorch and Tensorflow, and export datasets back into storage.

## Use Cases

*   **ETL:** Build Pythonic data pipelines to transform and enrich unstructured data, incorporating models, including LLMs.
*   **Analytics:** Analyze your data with a table-like API and vectorized engine, combining file and metadata for large-scale analysis.
*   **Versioning:** Manage large datasets without moving or copying data, ideal for buckets with millions of files.
*   **Incremental Processing:** Utilize delta and retry features for efficient workflows: process only new/changed data, and automatically retry records with errors.

## Getting Started

1.  **Installation:**

    ```bash
    pip install datachain
    ```

2.  **Explore the Documentation:**

    *   [Quick Start](https://docs.datachain.ai/quick-start)
    *   [Full Documentation](https://docs.datachain.ai/)

## Examples

### Download Subset of Files Based on Metadata

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

```python
import datachain as dc
from datachain import C, File

def process_file(file: File):
    try:
        content = file.read_text()
        result = analyze_content(content)
        return {
            "content": content,
            "result": result,
            "error": None
        }
    except Exception as e:
        return {
            "content": None,
            "result": None,
            "error": str(e)
        }

chain = (
    dc.read_storage(
        "data/",
        update=True,
        delta=True,
        delta_on="file.path",
        retry_on="error"
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

## Community and Support

*   [Documentation](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## Contributing

We welcome contributions!  Please see the [Contributor Guide](https://docs.datachain.ai/contributing) to learn more.

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams that offers:

*   **Centralized dataset registry** to manage data, code and dependencies in one place.
*   **Data Lineage** for data sources as well as derivative dataset.
*   **UI for Multimodal Data** like images, videos, and PDFs.
*   **Scalable Compute** to handle large datasets (100M+ files) and in-house AI model inference.
*   **Access control** including SSO and team based collaboration.