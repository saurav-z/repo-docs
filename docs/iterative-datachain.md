# DataChain: Build, Version, and Analyze Unstructured Data at Scale

**DataChain is a Python-based data warehouse designed to transform and analyze unstructured data like images, audio, videos, text, and PDFs, all without the hassle of data duplication.**  Visit the [DataChain GitHub Repository](https://github.com/iterative/datachain) for the latest updates.

## Key Features

*   **Multimodal Data Versioning:**
    *   Version unstructured data directly from cloud storage (S3, GCP, Azure, local file systems) without copying data.
    *   Supports diverse data types: images, video, text, PDFs, JSON, CSV, Parquet, and more.
    *   Combines files and metadata into persistent, versioned, and columnar datasets.
*   **Python-Native Functionality:**
    *   Operate on Python objects and their fields: float scores, strings, matrices, LLM responses, etc.
    *   Run Python code at scale on terabyte-sized datasets with built-in parallelization and memory efficiency. No SQL or Spark required.
*   **Advanced Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata; includes support for vector embedding searches.
    *   High-performance vectorized operations on Python objects: sum, count, average, etc.
    *   Integrates with PyTorch and TensorFlow, and supports exporting data back to storage.

## Use Cases

*   **ETL (Extract, Transform, Load):** A Pythonic framework for defining and executing unstructured data transformations and enrichments, including applying models like LLMs.
*   **Analytics:**  DataChain datasets provide a table-like API and a vectorized engine for scalable analytics.
*   **Versioning:**  DataChain manages data references without storing, moving, or copying data, perfect for large datasets.
*   **Incremental Processing:** Efficiently process large datasets using delta and retry features for handling new or changed files and error recovery.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Explore the [Quick Start](https://docs.datachain.ai/quick-start) and full [Documentation](https://docs.datachain.ai/) for comprehensive guides and examples.

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
        result = analyze_content(content)  # Placeholder function
        return {"content": content, "result": result, "error": None}
    except Exception as e:
        return {"content": None, "result": None, "error": str(e)}

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

### LLM-based Text File Evaluation

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
*   [File an Issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

For team collaboration and advanced features, consider [DataChain Studio](https://studio.datachain.ai/), which offers:

*   Centralized dataset registry.
*   Data lineage tracking.
*   UI for multimodal data.
*   Scalable compute capabilities.
*   Access control features.

## Contributing

We welcome contributions! Please see the [Contributor Guide](https://docs.datachain.ai/contributing) for details.