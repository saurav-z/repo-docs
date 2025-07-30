# DataChain: The AI-Data Warehouse for Unstructured Data

DataChain is a Python-based framework that transforms and analyzes unstructured data like images, audio, videos, text, and PDFs, enabling efficient data processing and analysis. ([Original Repo](https://github.com/iterative/datachain))

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Effortless Data Versioning**: Version unstructured data directly from cloud storage (S3, GCP, Azure) and local file systems without data duplication.
*   **Broad Data Support**: Works with a wide array of multimodal data formats, including images, videos, text, PDFs, JSON, CSV, and Parquet files.
*   **Python-Native**:  Operate on Python objects and object fields, enabling high-scale data operations with Python code, built-in parallelization and memory-efficient computing.
*   **Robust Data Enrichment and Processing**:
    *   Integrate with local AI models and LLM APIs for metadata generation.
    *   Filter, join, and group datasets by metadata. Perform vector embedding searches.
    *   Benefit from high-performance vectorized operations.
*   **Incremental Processing**: Utilize delta and retry features for efficient and reliable data processing workflows, including new data handling and error resolution.

## Use Cases

1.  **ETL (Extract, Transform, Load)**:  A Pythonic framework for managing unstructured data transformations, enrichments, and model applications.
2.  **Advanced Analytics**:  A dataframe-like API and a vectorized engine provide a powerful framework for large-scale data analysis.
3.  **Data Versioning**:  Achieve robust versioning without the need to store, move, or copy data.
4.  **Incremental Processing**: Employ delta and retry mechanisms for optimized processing of large datasets and error management.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Explore the `Quick Start <https://docs.datachain.ai/quick-start>`_ and `Docs <https://docs.datachain.ai/>`_ for comprehensive guidance.

## Examples

### Download a Subset of Files Based on Metadata

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

## Community and Support

*   `Docs <https://docs.datachain.ai/>`_
*   `File an issue <https://github.com/iterative/datachain/issues>`_
*   `Discord Chat <https://dvc.org/chat>`_
*   `Email <mailto:support@dvc.org>`_
*   `Twitter <https://twitter.com/DVCorg>`_

## DataChain Studio Platform

For advanced features and team collaboration, consider `DataChain Studio <https://studio.datachain.ai/>`, a proprietary platform offering:

*   Centralized dataset registry
*   Data lineage tracking
*   UI for multimodal data
*   Scalable compute for large datasets
*   Access control (SSO, team-based collaboration)