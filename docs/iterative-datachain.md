# DataChain: AI-Powered Data Warehouse for Unstructured Data

**Transform, analyze, and version your unstructured data with DataChain, a Python-based AI-data warehouse.**  For the original repo, see [iterative/datachain](https://github.com/iterative/datachain).

DataChain is a powerful tool designed to help you manage and process unstructured data efficiently. It integrates seamlessly with external storage, enabling you to work with large datasets without data duplication.

## Key Features:

*   📂 **Multimodal Data Versioning:**
    *   Version unstructured data (images, video, text, PDFs, etc.) without data copies by referencing cloud storage.
    *   Unite files and metadata into versioned, columnar datasets.
    *   Supports S3, GCP, Azure, and local file systems.
*   🐍 **Python-Friendly:**
    *   Operate on Python objects and object fields directly.
    *   Run Python code at scale, with built-in parallelization.
    *   No SQL or Spark required.
*   🧠 **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata. Search by vector embeddings.
    *   High-performance vectorized operations on Python objects.
    *   Integrates with PyTorch and TensorFlow.

## Use Cases

*   **ETL:** Pythonic framework for describing and running unstructured data transformations.
*   **Analytics:** Perform analytics on large-scale datasets with a dataframe-like API and vectorized engine.
*   **Versioning:** Version unstructured data (images, video, text, PDFs, etc.) without data copies.
*   **Incremental Processing:** Efficiently process only new or changed files with delta and retry features.

## Getting Started

For detailed instructions and examples, please visit the following resources:

*   [Quick Start](https://docs.datachain.ai/quick-start)
*   [Documentation](https://docs.datachain.ai/)

**Installation:**

```bash
pip install datachain
```

## Example: Download a Subset of Files Based on Metadata

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

We welcome contributions! Please see the [Contributor Guide](https://docs.datachain.ai/contributing) for details.

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution that provides:

*   **Centralized dataset registry** to manage data, code, and dependencies.
*   **Data Lineage** for data sources and derivative datasets.
*   **UI for Multimodal Data** like images, videos, and PDFs.
*   **Scalable Compute** for large datasets and in-house AI model inference.
*   **Access control** including SSO and team-based collaboration.