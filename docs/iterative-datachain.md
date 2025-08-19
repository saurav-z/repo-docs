# DataChain: Your Python-Powered AI-Data Warehouse for Unstructured Data

**DataChain** empowers you to transform, analyze, and version unstructured data like images, audio, videos, text, and PDFs with Python, without the need to move or copy your data. [Visit the original repository](https://github.com/iterative/datachain) for more details.

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Efficient Data Versioning:** Version your unstructured data without data duplication.
    *   Supports S3, GCP, Azure, and local file systems.
    *   Handles images, videos, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Combines files and metadata into persistent, versioned, columnar datasets.
*   **Pythonic Data Manipulation:** Leverage Python for all your data operations.
    *   Operate on Python objects and fields: scores, strings, matrices, LLM responses.
    *   High-scale processing on terabytes of data with built-in parallelization.
    *   No SQL or Spark expertise needed.
*   **Powerful Data Enrichment and Processing:** Integrate AI and LLMs into your workflows.
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata, with vector search capabilities.
    *   Vectorized operations for efficient data transformations.
    *   Seamless integration with PyTorch, TensorFlow, and external storage.
*   **Incremental Processing:** Efficiently handle evolving datasets with delta and retry features.
    *   Process only new or changed data (Delta Processing).
    *   Automatically retry failed operations (Retry Processing).
    *   Combine delta and retry for robust pipelines.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build and run unstructured data transformations, enrichments, and model applications using a Pythonic framework.
*   **Analytics:** Analyze data efficiently with a dataframe-like API and a vectorized engine on DataChain datasets.
*   **Data Versioning:** Manage versions of your data without data movement.
*   **Incremental Processing:** Process new data and fix errors in a single pipeline.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

For detailed instructions and examples, see the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/).

## Examples

### Download Subset of Files Based on Metadata

This example demonstrates how to download specific files based on metadata, like cat images with high confidence scores.

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

This example shows how to use delta and retry processing for efficient handling of large datasets.

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

This example shows how to evaluate chatbot conversations using LLM based evaluation.

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

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

DataChain Studio is a proprietary solution for teams, offering:

*   **Centralized dataset registry** to manage data, code, and dependencies.
*   **Data Lineage** for data sources and derivative datasets.
*   **UI for Multimodal Data** (images, videos, PDFs).
*   **Scalable Compute** for large datasets (100M+ files) and in-house AI model inference.
*   **Access control** including SSO and team-based collaboration.