# DataChain: AI-Powered Data Warehouse for Unstructured Data

**DataChain is a Python-based data processing framework designed to transform, analyze, and version unstructured data, making it easy to work with images, videos, text, and more.**  [Explore the source code on GitHub](https://github.com/iterative/datachain).

## Key Features

*   **Versioned Multimodal Datasets:**
    *   Version unstructured data (images, videos, text, PDFs, etc.) without data duplication, supporting S3, GCP, Azure, and local file systems.
    *   Unite files and metadata into persistent, versioned, columnar datasets.

*   **Python-Native Operations:**
    *   Operate on Python objects and fields (float scores, strings, matrices, LLM responses).
    *   Run Python code on large, terabyte-scale datasets with built-in parallelization and memory-efficient computing – no SQL or Spark needed.

*   **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata, including vector embedding searches.
    *   High-performance vectorized operations on Python objects (sum, count, average, etc.).
    *   Integrate seamlessly with PyTorch and TensorFlow; export data back to storage.

*   **Incremental Processing:**
    *   Process only new or changed files/records with delta processing.
    *   Retry processing for records with errors.
    *   Combine new data processing with error fixing in a single pipeline.

## Use Cases

1.  **ETL:** Pythonic framework for transforming and enriching unstructured data, including applying models like LLMs.
2.  **Analytics:** Analyze data at scale with a dataframe-like API and vectorized engine.
3.  **Versioning:** Version data without storing, moving, or copying data.
4.  **Incremental Processing:** Efficiently process evolving datasets using delta and retry features.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Explore the `Quick Start <https://docs.datachain.ai/quick-start>`_ and comprehensive `Docs <https://docs.datachain.ai/>`_ for detailed guidance.

## Examples

### Download Subset of Files Based on Metadata

Download a specific subset of files from cloud storage using metadata.

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

Process large datasets efficiently with delta and retry features.

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

### LLM-based Text-File Evaluation

Evaluate chatbot conversations using an LLM.

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

## Contributing

Contributions are welcome; consult the `Contributor Guide <https://docs.datachain.ai/contributing>`_.

## DataChain Studio Platform

`DataChain Studio <https://studio.datachain.ai/>`_ provides a proprietary solution for teams, including:

*   Centralized dataset registry
*   Data Lineage
*   UI for Multimodal Data
*   Scalable Compute
*   Access control (SSO, team-based collaboration)