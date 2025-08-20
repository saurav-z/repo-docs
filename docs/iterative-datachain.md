# DataChain: Your Python-Powered AI Data Warehouse for Unstructured Data

DataChain is a Python-based AI data warehouse that simplifies transforming, analyzing, and versioning unstructured data like images, audio, videos, text, and PDFs.  [Learn more on GitHub](https://github.com/iterative/datachain).

## Key Features

*   **Version Control for Unstructured Data:** Efficiently version unstructured data without data duplication. Supports S3, GCP, Azure, and local file systems.
*   **Multimodal Data Support:** Seamlessly handles various data types, including images, videos, text, PDFs, JSONs, CSVs, and Parquet.
*   **Python-Native:** Work with Python objects and fields, run Python code at scale with built-in parallelization.
*   **Data Enrichment & Processing:** Generate metadata, filter, join, and group datasets. Perform high-performance vectorized operations.
*   **Incremental Processing:**  Utilizes delta and retry features for efficient workflows, handling new or changed files, retrying errors.
*   **Integration with LLMs:** Easily evaluate and process data with LLMs.

## Use Cases

*   **ETL (Extract, Transform, Load):** Pythonic framework for unstructured data transformations and enrichment, including applying models like LLMs.
*   **Analytics:** Analyze data with a dataframe-like API and vectorized engine for efficient, large-scale analysis.
*   **Versioning:** Version your data without the need to move or copy your data. Works great on buckets with millions of files!
*   **Incremental Processing:** Process only new or changed files, or automatically reprocess records with errors for efficient processing.

## Getting Started

Explore the [Quick Start](https://docs.datachain.ai/quick-start) and the comprehensive [Documentation](https://docs.datachain.ai/) to begin using DataChain.

```bash
pip install datachain
```

## Examples

**(Examples from original README, slightly improved)**

### Example: Download Subset of Files Based on Metadata

Filter and download specific files from cloud storage using metadata.

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

### Example: Incremental Processing with Error Handling

Process large datasets efficiently with delta and retry processing.

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

### Example: LLM Based Text File Evaluation

Evaluate chatbot conversations using LLMs.

```shell
$ pip install mistralai  # Requires version >=1.0.0
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

This example successfully processed 31 out of 50 files.

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution offering:

*   Centralized dataset registry
*   Data Lineage
*   UI for Multimodal Data
*   Scalable Compute
*   Access control