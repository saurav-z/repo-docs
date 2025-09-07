# DataChain: Your Python-Powered AI Data Warehouse for Unstructured Data

**DataChain** simplifies the transformation and analysis of unstructured data like images, audio, video, text, and PDFs, providing a Python-native solution for efficient and scalable data workflows. [View the project on GitHub](https://github.com/iterative/datachain).

## Key Features

*   **Multimodal Data Handling:**
    *   Effortlessly version unstructured data without data duplication.
    *   Supports various data types: images, video, text, PDFs, JSON, CSV, Parquet, and more.
    *   Combines files and metadata into persistent, versioned, columnar datasets.

*   **Pythonic Interface:**
    *   Work directly with Python objects and their fields (float scores, strings, matrices, LLM responses).
    *   Execute Python code on large-scale datasets with built-in parallelization and memory-efficient computing, eliminating the need for SQL or Spark.

*   **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata.
    *   Perform high-performance vectorized operations on Python objects.
    *   Seamlessly integrate with PyTorch and TensorFlow, or export data back to storage.

*   **ETL Capabilities:** Utilize a Pythonic framework to describe and execute unstructured data transformations, enrichments, and model application, including LLMs.

*   **Data Analytics:** Analyze DataChain datasets using a table-like API and vectorized engine at scale.

*   **Incremental Processing:** Implement efficient workflows with delta and retry features:
    *   **Delta Processing:** Process only new or changed files/records.
    *   **Retry Processing:** Automatically reprocess records with errors or missing results.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build data pipelines for unstructured data.
*   **Data Analytics:** Perform scalable analysis on combined file and metadata.
*   **Data Versioning:** Manage large datasets without the need to copy or move data.

## Getting Started

Explore the `Quick Start <https://docs.datachain.ai/quick-start>`_ and comprehensive `Docs <https://docs.datachain.ai/>`_ to begin using DataChain.

Install DataChain:

```bash
pip install datachain
```

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

## Example: LLM Based Text-File Evaluation

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

For advanced features and team collaboration, consider `DataChain Studio <https://studio.datachain.ai/>`, offering:

*   Centralized dataset registry
*   Data lineage visualization
*   UI for multimodal data
*   Scalable compute for large datasets
*   Access control