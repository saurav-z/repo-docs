# DataChain: The Python-Powered AI Data Warehouse for Unstructured Data

**DataChain** empowers you to transform, analyze, and version unstructured data like images, audio, video, text, and PDFs with ease. Visit the [DataChain GitHub Repository](https://github.com/iterative/datachain) for more information.

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Effortless Data Versioning:**
    *   Version unstructured data without data duplication.
    *   Supports various storage locations: S3, GCP, Azure, and local file systems.
    *   Handles various data types: images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unites files and metadata into versioned columnar datasets.

*   **Pythonic Data Handling:**
    *   Works seamlessly with Python objects and fields: float scores, strings, matrixes, LLM response objects.
    *   Run Python code on terabytes-sized datasets, with built-in parallelization and memory efficiency.
    *   No need for SQL or Spark.

*   **Advanced Data Enrichment and Processing:**
    *   Integrates local AI models and LLM APIs for metadata generation.
    *   Offers advanced filtering, joining, and grouping capabilities by metadata.
    *   Enables search through vector embeddings.
    *   Provides high-performance vectorized operations on Python objects (sum, count, avg, etc.).
    *   Seamlessly integrates with Pytorch and Tensorflow, and allows data export back to storage.

## Use Cases

1.  **ETL (Extract, Transform, Load):** Pythonic framework for data transformation and enrichment, including applying models.
2.  **Analytics:** Analyze data using a dataframe-like API and vectorized engine for large-scale analytics.
3.  **Versioning:** Version data without moving or copying data. Ideal for large datasets like image or video buckets.
4.  **Incremental Processing:** Utilize delta processing and retry features for efficient data pipelines.

## Getting Started

Install DataChain with pip:

```bash
pip install datachain
```

Explore the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/) for detailed information.

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

### LLM-Based Text File Evaluation

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

*   [Documentation](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams, offering:

*   Centralized dataset registry
*   Data Lineage
*   UI for Multimodal Data
*   Scalable Compute
*   Access control