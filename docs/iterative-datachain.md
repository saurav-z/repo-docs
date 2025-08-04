# DataChain: Effortlessly Transform and Analyze Unstructured Data

DataChain is a Python-based AI data warehouse that empowers you to efficiently process and analyze unstructured data like images, videos, text, and PDFs.  [Explore the original repository](https://github.com/iterative/datachain).

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Multimodal Data Support & Versioning:**
    *   Effortlessly version unstructured data (images, video, text, PDFs, JSON, CSV, parquet, etc.) without data duplication, supporting references to S3, GCP, Azure, and local file systems.
    *   Unite files and metadata into persistent, versioned, columnar datasets.
*   **Python-Native Integration:**
    *   Operate directly on Python objects and their fields, including float scores, strings, matrixes, and LLM response objects.
    *   Run high-scale operations (terabytes of data) with built-in parallelization and efficient memory usage, eliminating the need for SQL or Spark.
*   **AI-Powered Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata; search by vector embeddings.
    *   Perform high-performance vectorized operations on Python objects: sum, count, average, etc.
    *   Seamlessly integrate with PyTorch and TensorFlow, or export data back into storage.
*   **ETL, Analytics, and Incremental Processing:**
    *   Pythonic framework for describing and running unstructured data transformations and enrichments.
    *   DataChain datasets combine all data object information in one place, providing a dataframe-like API and vectorized engine for large-scale analytics.
    *   Delta and retry features for efficient processing workflows:
        *   **Delta Processing:** Process only new or changed files/records.
        *   **Retry Processing:** Automatically reprocess records with errors.
        *   **Combined Approach:** Process new data and fix errors in a single pipeline.

## Use Cases

1.  **ETL**: Pythonic framework for describing and running unstructured data transformations and enrichments, applying models to data, including LLMs.
2.  **Analytics**: DataChain datasets combine all data object information in one place, providing a dataframe-like API and vectorized engine for large-scale analytics.
3.  **Versioning**: Supports versioning of unstructured data without moving or creating data copies, by supporting references to S3, GCP, Azure, and local file systems.
4.  **Incremental Processing**: Delta and retry features allow for efficient processing workflows.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

Visit the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/) for detailed information.

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

### LLM-Based Text-File Evaluation

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

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution for teams, offering:

*   **Centralized dataset registry** to manage data, code, and dependencies.
*   **Data Lineage** for data sources and derivative datasets.
*   **UI for Multimodal Data** like images, videos, and PDFs.
*   **Scalable Compute** to handle large datasets (100M+ files) and in-house AI model inference.
*   **Access control** including SSO and team-based collaboration.