<div align="center">
  <img src="docs/assets/datachain.svg" alt="DataChain Logo" width="100"/>
  <h1>DataChain: AI-Powered Data Warehouse for Unstructured Data</h1>
</div>

DataChain empowers you to transform, analyze, and version unstructured data, from images to text, all within a Python-friendly framework.  [Explore the DataChain Repository](https://github.com/iterative/datachain)

## Key Features

*   **Multimodal Dataset Versioning:**
    *   Version unstructured data without data duplication, working seamlessly with cloud storage (S3, GCP, Azure) and local files.
    *   Supports a wide variety of data types: images, video, text, PDFs, JSONs, CSVs, Parquet, and more.
    *   Creates persistent, versioned, columnar datasets by combining files and metadata.

*   **Python-Friendly:**
    *   Operate directly on Python objects and their fields, including scores, strings, matrixes, and LLM responses.
    *   Run Python code efficiently on terabyte-scale datasets, with built-in parallelization for high performance.
    *   No SQL or Spark expertise required.

*   **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata and perform vector-based searches.
    *   Utilize high-performance vectorized operations on Python objects (sum, count, average, etc.).
    *   Seamlessly integrate with PyTorch and TensorFlow, and easily export data back into storage.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build Pythonic pipelines for transforming and enriching unstructured data, including the application of models (like LLMs).
*   **Analytics:** Analyze data at scale using DataChain datasets, which provide a table-like API with a vectorized engine.
*   **Versioning:** Efficiently manage versions of unstructured data without requiring data duplication (ideal for large datasets of images, videos, etc.).
*   **Incremental Processing:** Leverage delta and retry features for efficient data processing workflows:
    *   **Delta Processing:** Process only new or modified files.
    *   **Retry Processing:** Automatically reprocess records that encountered errors.
    *   **Combined Approach:** Process new data and fix errors in a single pipeline.

## Getting Started

*   **Install:**

```bash
pip install datachain
```

*   **Quick Start:** [Quick Start Guide](https://docs.datachain.ai/quick-start)
*   **Documentation:** [DataChain Documentation](https://docs.datachain.ai/)

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

*   **Documentation:** [DataChain Documentation](https://docs.datachain.ai/)
*   **File an issue:** [GitHub Issues](https://github.com/iterative/datachain/issues)
*   **Discord Chat:** [DVC Discord](https://dvc.org/chat)
*   **Email:** [support@dvc.org](mailto:support@dvc.org)
*   **Twitter:** [@DVCorg](https://twitter.com/DVCorg)

## DataChain Studio Platform

`DataChain Studio` is a commercial platform offering:

*   Centralized dataset registry.
*   Data lineage.
*   UI for multimodal data.
*   Scalable compute.
*   Access control and collaboration.