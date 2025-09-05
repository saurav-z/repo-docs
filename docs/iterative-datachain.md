# DataChain: Unleash the Power of Your Unstructured Data

**DataChain is a Python-based AI-data warehouse designed to transform, analyze, and version unstructured data like images, audio, video, text, and PDFs at scale.** [Visit the DataChain GitHub Repository](https://github.com/iterative/datachain)

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Multimodal Data Support:** Process and version images, videos, text, PDFs, JSONs, CSVs, and parquet files.
*   **Version Control for Unstructured Data:** Manage unstructured data without data duplication, using references to storage like S3, GCP, Azure, and local file systems.
*   **Python-First Development:** Utilize a Python-friendly framework with dataframe-like APIs, vectorized engine, and built-in parallelization.
*   **Efficient Data Processing:** Includes delta processing (process only new/changed files) and retry processing (automatically reprocess records with errors).
*   **Metadata Enrichment and Analytics:** Generate, filter, join, and group datasets by metadata.  Perform high-performance vectorized operations.
*   **LLM Integration:** Seamlessly integrates with Large Language Models (LLMs) for text analysis, and other AI-driven data enrichments.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build Pythonic pipelines for transforming and enriching unstructured data.
*   **Analytics:** Perform analytics on data at scale using a dataframe-like API.
*   **Data Versioning:** Manage versions of your data without moving or copying files.
*   **Incremental Processing:** Efficiently process new and changed data with built-in delta and retry features.

## Getting Started

1.  **Installation:**

    ```bash
    pip install datachain
    ```

2.  **Quick Start and Documentation:**

    *   [Quick Start](https://docs.datachain.ai/quick-start)
    *   [Documentation](https://docs.datachain.ai/)

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

## Example: LLM based text-file evaluation

```shell
$ pip install mistralai # Requires version >=1.0.0
$ export MISTRAL_API_KEY=_your_key_
```

Python code:

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

With the instruction above, the Mistral model considers 31/50 files to hold the successful dialogues:

```shell
$ ls output_mistral/datachain-demo/chatbot-KiT/
1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
$ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
31
```

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams that offers:

*   **Centralized dataset registry** to manage data, code and dependencies in one place.
*   **Data Lineage** for data sources as well as derivative dataset.
*   **UI for Multimodal Data** like images, videos, and PDFs.
*   **Scalable Compute** to handle large datasets (100M+ files) and in-house AI model inference.
*   **Access control** including SSO and team based collaboration.