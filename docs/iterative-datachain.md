# DataChain: Transform, Analyze, and Version Your Unstructured Data

**DataChain is a Python-based data warehouse designed to streamline the transformation, analysis, and versioning of unstructured data, such as images, audio, videos, text, and PDFs.** [View the original repository](https://github.com/iterative/datachain)

## Key Features

*   **Multimodal Data Support:**
    *   Version control unstructured data efficiently without data duplication, supporting references to various storage locations (S3, GCP, Azure, local file systems).
    *   Processes a wide variety of data formats including images, video, text, PDFs, JSONs, CSVs, and parquet.
    *   Combines files and metadata into persistent, versioned, and columnar datasets.

*   **Pythonic Interface:**
    *   Operate directly on Python objects and their attributes, including scores, strings, matrixes, and LLM response objects.
    *   Run Python code at scale on terabytes-sized datasets, with built-in parallelization and optimized memory usage, eliminating the need for SQL or Spark.

*   **Advanced Data Processing and Enrichment:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata. Supports vector embedding search.
    *   Execute high-performance vectorized operations on Python objects, such as sum, count, and average.
    *   Seamlessly integrate datasets with PyTorch and TensorFlow, and export them back into storage.

## Use Cases

1.  **ETL:** A Pythonic framework for describing and running unstructured data transformations, enrichment, and model application (including LLMs).
2.  **Analytics:** DataChain datasets combine all information about data objects, and provide a DataFrame-like API with a vectorized engine for large-scale analytics.
3.  **Versioning:** Manage your unstructured data without the need to copy or move data. Ideal for large datasets stored in cloud buckets (millions of images, videos, etc.).
4.  **Incremental Processing:**  DataChain's delta and retry features enable efficient processing workflows:

    *   **Delta Processing:** Process only new or changed files/records.
    *   **Retry Processing:** Automatically reprocess records with errors or missing results.
    *   **Combined Approach:** Process new data and fix errors in a single pipeline.

## Getting Started

*   Visit the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/) to get started.
*   Install DataChain:

    ```bash
    pip install datachain
    ```

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

### LLM Based Text-File Evaluation

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

```shell
$ ls output_mistral/datachain-demo/chatbot-KiT/
1.txt  15.txt 18.txt 2.txt  22.txt 25.txt 28.txt 33.txt 37.txt 4.txt  41.txt ...
$ ls output_mistral/datachain-demo/chatbot-KiT/ | wc -l
31
```

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams offering:

*   Centralized dataset registry.
*   Data Lineage.
*   UI for Multimodal Data.
*   Scalable Compute.
*   Access control (SSO and team collaboration).