# DataChain: AI-Powered Data Warehouse for Unstructured Data

**DataChain is a Python-based data warehouse designed for transforming, analyzing, and versioning unstructured data, making AI-driven insights accessible and efficient.** ([Original Repository](https://github.com/iterative/datachain))

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   ✅ **Effortless Unstructured Data Management**: Version unstructured data without data duplication by referencing data in S3, GCP, Azure, or local file systems. Supports various data types including images, videos, text, PDFs, JSONs, CSVs, and Parquet files.
*   ✅ **Pythonic Framework**: Enjoy a Python-friendly API for transforming, analyzing, and enriching data. Operate on Python objects, utilize built-in parallelization, and perform memory-efficient computations without needing SQL or Spark.
*   ✅ **Powerful Data Processing**: Enrich data with metadata using local AI models and LLM APIs, perform filtering, joining, and grouping operations based on metadata, and search using vector embeddings. Supports high-performance vectorized operations on Python objects.
*   ✅ **Incremental Processing**:  Leverage delta and retry features to efficiently process only new/changed files, automatically retry processing errors, and combine both strategies in a single pipeline.
*   ✅ **DataChain Studio Integration:** Seamless integration with the DataChain Studio platform for centralized dataset management, data lineage tracking, UI support for multimodal data, scalable compute, and access control features.

## Use Cases

*   **ETL:** Pythonic framework for describing and running unstructured data transformations
   and enrichments, applying models to data, including LLMs.
*   **Analytics:** DataChain dataset is a table that combines all the information about data
   objects in one place + it provides dataframe-like API and vectorized engine to do analytics
   on these tables at scale.
*   **Versioning:** DataChain doesn't store, require moving or copying data (unlike DVC).
   Perfect use case is a bucket with thousands or millions of images, videos, audio, PDFs.
*   **Incremental Processing:** DataChain's delta and retry features allow for efficient
   processing workflows.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

Explore the `Quick Start <https://docs.datachain.ai/quick-start>`_ and `Docs <https://docs.datachain.ai/>`_ to learn more.

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

Python code:
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

*   `Docs <https://docs.datachain.ai/>`_
*   `File an issue <https://github.com/iterative/datachain/issues>`_
*   `Discord Chat <https://dvc.org/chat>`_
*   `Email <mailto:support@dvc.org>`_
*   `Twitter <https://twitter.com/DVCorg>`_

## DataChain Studio Platform

`DataChain Studio <https://studio.datachain.ai/>` is a proprietary solution for teams that offers:

*   **Centralized dataset registry** to manage data, code, and dependencies in one place.
*   **Data Lineage** for data sources as well as derivative dataset.
*   **UI for Multimodal Data** like images, videos, and PDFs.
*   **Scalable Compute** to handle large datasets (100M+ files) and in-house AI model inference.
*   **Access control** including SSO and team-based collaboration.

## Contributing

Contributions are welcome. Refer to the `Contributor Guide <https://docs.datachain.ai/contributing>`_ for details.