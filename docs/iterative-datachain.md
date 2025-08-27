<div align="center">
  <img src="docs/assets/datachain.svg" alt="DataChain Logo" width="100"/>
  <h1>DataChain: Your Python-Powered AI Data Warehouse</h1>
</div>

**DataChain is a powerful Python library designed for transforming, analyzing, and versioning unstructured data, such as images, audio, video, text, and PDFs, enabling you to build efficient AI data pipelines.**

[Go to the original repository](https://github.com/iterative/datachain)

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Effortless Data Versioning:**
    *   Version unstructured data without data duplication, supporting S3, GCP, Azure, and local file systems.
    *   Handle multimodal data: images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unite files and metadata into persistent, versioned, columnar datasets.

*   **Pythonic Data Workflows:**
    *   Operate on Python objects and object fields such as float scores, strings, matrixes, and LLM response objects.
    *   Run Python code on large-scale datasets with built-in parallelization.
    *   No SQL or Spark required.

*   **Advanced Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata and search by vector embeddings.
    *   Vectorized operations for high-performance data analysis on Python objects.
    *   Integrate with Pytorch and Tensorflow and export datasets back to storage.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build Pythonic pipelines for transforming and enriching unstructured data, including applying models like LLMs.
*   **Analytics:** Analyze data efficiently with a table-like API and a vectorized engine for large-scale datasets.
*   **Versioning:** Efficiently version unstructured data in cloud storage (e.g., thousands or millions of images).
*   **Incremental Processing:** Utilize delta and retry features for efficient data processing, including processing new or changed data, and reprocessing records with errors.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Explore the [Quick Start](https://docs.datachain.ai/quick-start) and the full [Docs](https://docs.datachain.ai/) to get started and learn more.

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

## Example: LLM based text-file evaluation

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

Contributions are welcome! Refer to the [Contributor Guide](https://docs.datachain.ai/contributing) for more information.

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams that offers:

*   **Centralized dataset registry** to manage data, code, and dependencies.
*   **Data Lineage** for data sources and derivative datasets.
*   **UI for Multimodal Data** like images, videos, and PDFs.
*   **Scalable Compute** to handle large datasets and in-house AI model inference.
*   **Access control** including SSO and team-based collaboration.