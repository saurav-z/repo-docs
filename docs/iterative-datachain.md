# DataChain: Build AI-Powered Data Pipelines for Unstructured Data

**DataChain is a Python-based framework that simplifies the transformation, analysis, and versioning of unstructured data like images, audio, videos, and text.** [Explore the DataChain repository on GitHub](https://github.com/iterative/datachain).

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Effortless Versioning:** Version and manage unstructured data without data duplication, supporting data from S3, GCP, Azure, and local file systems.
*   **Multimodal Data Support:** Works with diverse data formats including images, video, text, PDFs, JSONs, CSVs, and parquet.
*   **Pythonic Interface:** Built with Python, enabling seamless integration with your existing workflows, and high-performance vectorized operations on Python objects.
*   **Data Enrichment and Processing:** Enrich data using local AI models, LLM APIs, or built-in vectorized operations, and filter/join/group datasets efficiently.
*   **Incremental Processing:**  Process only new or changed files with delta processing, retry features, and error handling.

## Use Cases

*   **ETL:** Build data pipelines to transform and enrich unstructured data.
*   **Analytics:** Perform scalable analytics on combined data and metadata.
*   **Versioning:** Version your datasets without data duplication or copying.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Dive into the details with the [Quick Start](https://docs.datachain.ai/quick-start) and the full [Docs](https://docs.datachain.ai/).

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

Contributions are very welcome. See the [Contributor Guide](https://docs.datachain.ai/contributing) to learn more.

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams, offering:

*   Centralized dataset registry.
*   Data Lineage.
*   UI for Multimodal Data.
*   Scalable Compute.
*   Access control.