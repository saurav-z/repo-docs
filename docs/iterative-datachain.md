# DataChain: Build AI-Powered Data Warehouses for Unstructured Data

**DataChain** is a powerful Python-based framework for transforming, analyzing, and versioning unstructured data, providing an efficient and scalable solution for modern data challenges. [View the original repository](https://github.com/iterative/datachain).

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Unstructured Data Support:** Handles images, audio, videos, text, PDFs, and more.
*   **Versioning:** Efficiently version data without copying, supporting S3, GCP, Azure, and local file systems.
*   **Pythonic API:**  Provides a user-friendly Python interface with support for high-scale datasets.
*   **Metadata Management:**  Enables easy querying and efficient data retrieval with metadata stored in an internal database.
*   **ETL Capabilities:** Pythonic framework for describing and running unstructured data transformations.
*   **Analytics:** Combines all information about data objects in one place + provides dataframe-like API and vectorized engine to do analytics on these tables at scale.
*   **Incremental Processing:** Supports delta and retry features for efficient workflows, including:
    *   **Delta Processing:** Process only new or changed files.
    *   **Retry Processing:** Automatically reprocess records with errors.
    *   **Combined Approach:** Process new data and fix errors in a single pipeline.
*   **Data Enrichment & Processing:** Generate and utilize metadata, filter, join, group, and search by vector embeddings.
*   **Scalable Performance:** High-performance vectorized operations on Python objects for optimized data processing.

## Use Cases

*   **ETL (Extract, Transform, Load):**  Transform and enrich unstructured data using Python.
*   **Analytics:** Perform scalable data analysis on large, unstructured datasets.
*   **Versioning:** Manage and version large datasets of unstructured files without data duplication.
*   **Incremental Processing:** Streamline data processing pipelines with delta and retry features.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Explore the `Quick Start <https://docs.datachain.ai/quick-start>`_ and comprehensive `Docs <https://docs.datachain.ai/>`_ for detailed information and examples.

## Example: Download Subset of Files Based on Metadata

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

## Example: Incremental Processing with Error Handling

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

## Example: LLM Based Text-File Evaluation

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

*   `Docs <https://docs.datachain.ai/>`_
*   `File an issue`_
*   `Discord Chat <https://dvc.org/chat>`_
*   `Email <mailto:support@dvc.org>`_
*   `Twitter <https://twitter.com/DVCorg>`_

## DataChain Studio Platform

Explore `DataChain Studio <https://studio.datachain.ai/>`_ for a proprietary solution that offers a centralized dataset registry, data lineage, a UI for multimodal data, scalable compute, and access control.

## Contributing

Contributions are welcome! See the `Contributor Guide`_.