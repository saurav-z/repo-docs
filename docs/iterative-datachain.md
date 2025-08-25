# DataChain: The Python-Powered AI Data Warehouse for Unstructured Data

**DataChain** is a Python-based solution designed to transform, analyze, and version unstructured data like images, audio, videos, and text, enabling efficient data workflows.  [Explore the DataChain Repository](https://github.com/iterative/datachain).

**Key Features:**

*   **🔄 Versatile Data Versioning:**
    *   Version unstructured data without data duplication, using references to S3, GCP, Azure, and local files.
    *   Supports a wide variety of data types: images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unite files and metadata into persistent, versioned, columnar datasets.
*   **🐍 Pythonic Approach:**
    *   Operate directly on Python objects and their fields (float scores, strings, matrices, LLM responses).
    *   Run Python code at scale on terabyte-sized datasets with built-in parallelization and memory-efficient computing – no SQL or Spark required.
*   **🧠 Powerful Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata; search by vector embeddings.
    *   High-performance vectorized operations on Python objects (sum, count, average, etc.).
    *   Seamlessly integrate with PyTorch and TensorFlow, and export datasets back into storage.
*   **⚙️ ETL, Analytics, and Incremental Processing:**
    *   Pythonic framework for transforming and enriching unstructured data.
    *   DataChain datasets combine file and metadata information.
    *   Delta and retry features for efficient processing.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build and run Pythonic data transformation pipelines, applying models to your data, including LLMs.
*   **Analytics:** Analyze your datasets at scale with a dataframe-like API and vectorized engine.
*   **Versioning:** Easily manage and version large datasets without data duplication.
*   **Incremental Processing:** Process only new or changed files with delta and retry features.

## Getting Started

*   **Installation:**

    ```bash
    pip install datachain
    ```

*   **Documentation:**

    *   Visit the [Quick Start Guide](https://docs.datachain.ai/quick-start) and the full [Documentation](https://docs.datachain.ai/) to begin using `DataChain`.

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
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## Contributing

Contributions are welcome! See the [Contributor Guide](https://docs.datachain.ai/contributing) for more details.

## DataChain Studio Platform

DataChain also offers a proprietary solution for teams, [DataChain Studio](https://studio.datachain.ai/), which provides:

*   Centralized dataset registry
*   Data lineage
*   UI for multimodal data
*   Scalable compute
*   Access control