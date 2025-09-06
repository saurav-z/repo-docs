# DataChain: AI-Powered Data Warehouse for Unstructured Data

**DataChain is a Python-based data warehouse designed for transforming, analyzing, and versioning unstructured data, such as images, audio, videos, text, and PDFs.** [Check out the original repo](https://github.com/iterative/datachain).

## Key Features

*   **Multimodal Data Support:**
    *   Version unstructured data without data duplication, working with references to various storage options (S3, GCP, Azure, local).
    *   Supports images, videos, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Unites files and metadata into persistent, versioned, columnar datasets.
*   **Pythonic Framework:**
    *   Operate on Python objects and their fields (float scores, strings, matrices, LLM response objects).
    *   Run Python code on large datasets with built-in parallelization and memory-efficient computing; no SQL or Spark needed.
*   **Data Enrichment and Processing:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata; search by vector embeddings.
    *   High-performance vectorized operations on Python objects (sum, count, avg, etc.).
    *   Seamlessly integrate with PyTorch and TensorFlow, or export data back into storage.
*   **ETL Capabilities:**
    *   Pythonic framework for describing and running unstructured data transformations and enrichments.
    *   Apply models to data, including LLMs.
*   **Analytics and Versioning:**
    *   DataChain dataset is a table that combines all the information about data objects in one place.
    *   Provides dataframe-like API and vectorized engine to do analytics on these tables at scale.
    *   No data moving or copying is needed, making it ideal for large datasets.
    *   Supports incremental processing with delta and retry features.

## Getting Started

*   **Installation:**

    ```bash
    pip install datachain
    ```

*   **Quick Start:**  Visit the [Quick Start Guide](https://docs.datachain.ai/quick-start) and [Documentation](https://docs.datachain.ai/) to begin using DataChain.

## Examples

### Downloading a Subset of Files Based on Metadata

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

### LLM-Based Text File Evaluation

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

## Contributing

Contributions are welcome! Review the [Contributor Guide](https://docs.datachain.ai/contributing) for more information.

## Community and Support

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution that offers:

*   Centralized dataset registry for managing data, code, and dependencies.
*   Data lineage for data sources and derivative datasets.
*   UI for Multimodal Data (images, videos, PDFs).
*   Scalable Compute to handle large datasets (100M+ files) and in-house AI model inference.
*   Access control, including SSO and team-based collaboration.