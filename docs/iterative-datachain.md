# DataChain: Your Python-Powered AI Data Warehouse for Unstructured Data

**DataChain** transforms and analyzes unstructured data like images, audio, video, text, and PDFs, making it easier to build AI-powered applications. Learn more on [GitHub](https://github.com/iterative/datachain).

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features:

*   **Versioned Multimodal Data:**
    *   Version unstructured data without data duplication.
    *   Supports images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unites files and metadata into persistent, versioned, columnar datasets.

*   **Pythonic and User-Friendly:**
    *   Operate on Python objects and object fields (float scores, strings, matrixes).
    *   Run Python code on terabyte-scale datasets with built-in parallelization and memory efficiency.
    *   No SQL or Spark is required.

*   **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata; search by vector embeddings.
    *   High-performance vectorized operations (sum, count, avg, etc.) on Python objects.
    *   Integrate with Pytorch and Tensorflow, or export data back into storage.

## Use Cases

*   **ETL:** Pythonic framework for describing and running unstructured data transformations and enrichments.
*   **Analytics:** DataChain datasets offer dataframe-like APIs for large-scale analytics.
*   **Versioning:** Version unstructured data without moving or creating data copies.
*   **Incremental Processing:** Efficient processing workflows with delta and retry features.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

For detailed documentation and quick start guides, please visit the [DataChain Documentation](https://docs.datachain.ai/).

## Example: Downloading a Subset of Files Based on Metadata

```python
import datachain as dc

meta = dc.read_json("gs://datachain-demo/dogs-and-cats/*json", column="meta", anon=True)
images = dc.read_storage("gs://datachain-demo/dogs-and-cats/*jpg", anon=True)

images_id = images.map(id=lambda file: file.path.split('.')[-2])
annotated = images_id.merge(meta, on="id", right_on="meta.id")

likely_cats = annotated.filter((dc.Column("meta.inference.confidence") > 0.93)
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

## Example: LLM-Based Text File Evaluation

```shell
$ pip install mistralai # Requires version >=1.0.0
$ export MISTRAL_API_KEY=_your_key_
```

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

## Contributing

We welcome contributions! Please see the [Contributor Guide](https://docs.datachain.ai/contributing) for more information.

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution for teams, offering:

*   Centralized dataset registry.
*   Data Lineage.
*   UI for Multimodal Data.
*   Scalable Compute.
*   Access control.