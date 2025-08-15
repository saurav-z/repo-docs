# DataChain: Transform and Analyze Unstructured Data with Ease

**DataChain** is a Python-based AI-data warehouse that simplifies the transformation, analysis, and versioning of unstructured data like images, audio, videos, text, and PDFs.  [Explore the DataChain GitHub repository](https://github.com/iterative/datachain).

## Key Features

*   **Multimodal Dataset Versioning:**
    *   Version unstructured data without data duplication, supporting references to S3, GCP, Azure, and local file systems.
    *   Handles diverse data types: images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unites files and metadata into persistent, versioned, columnar datasets.

*   **Python-Friendly:**
    *   Operate directly on Python objects and object fields: float scores, strings, matrixes, LLM response objects, etc.
    *   Run Python code at scale on terabytes-sized datasets, with built-in parallelization and memory-efficient computing.
    *   No SQL or Spark required.

*   **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata. Search by vector embeddings.
    *   High-performance vectorized operations on Python objects: sum, count, avg, etc.
    *   Seamlessly integrate datasets with PyTorch and TensorFlow, or export back into storage.

## Use Cases

*   **ETL:** Pythonic framework for describing and running unstructured data transformations and enrichments, including LLMs.
*   **Analytics:** Analyze data with a dataframe-like API and vectorized engine for efficient analytics at scale.
*   **Versioning:** Version data without moving or copying it, ideal for large datasets stored in object storage.
*   **Incremental Processing:** Delta processing, retry features, and combined approaches for efficient workflows.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

For detailed instructions and examples, visit the [Quick Start](https://docs.datachain.ai/quick-start) and [Documentation](https://docs.datachain.ai/).

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

### LLM based text-file evaluation

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

DataChain Studio is a proprietary solution for teams that offers:

*   Centralized dataset registry to manage data, code, and dependencies in one place.
*   Data Lineage for data sources and derivative datasets.
*   UI for Multimodal Data like images, videos, and PDFs.
*   Scalable Compute to handle large datasets (100M+ files) and in-house AI model inference.
*   Access control including SSO and team-based collaboration.