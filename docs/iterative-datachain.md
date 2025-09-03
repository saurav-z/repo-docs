# DataChain: Transform and Analyze Unstructured Data with Ease

**DataChain is a Python-based AI-data warehouse designed to efficiently process and analyze unstructured data like images, audio, videos, text, and PDFs.**

[Explore the DataChain Repository](https://github.com/iterative/datachain)

## Key Features

*   **Version Control for Unstructured Data:**
    *   Version your unstructured data without data duplication by referencing files in cloud storage (S3, GCP, Azure) and local file systems.
    *   Supports various data types: images, video, text, PDFs, JSONs, CSVs, parquet, and more.
    *   Combines files and metadata into persistent, versioned, and columnar datasets.

*   **Python-Native Data Processing:**
    *   Works directly with Python objects and their fields: float scores, strings, matrices, and LLM response objects.
    *   Processes terabytes of data at scale using built-in parallelization and memory-efficient computing, without SQL or Spark.

*   **Data Enrichment and Transformation:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata. Perform vector-based search.
    *   Leverage high-performance vectorized operations on Python objects: sum, count, average, etc.
    *   Integrate with PyTorch and TensorFlow or export to storage.

## Use Cases

*   **ETL (Extract, Transform, Load):** Build Pythonic frameworks for unstructured data transformations and enrichments, and integrate with LLMs.
*   **Analytics:** Analyze your data at scale using a dataframe-like API and vectorized engine. DataChain datasets unify files and metadata.
*   **Versioning:** Version your data without storing, moving, or copying, perfect for large datasets like images, videos, audio, and PDFs.
*   **Incremental Processing:** Efficiently process data with delta processing, retries, and a combined approach to handle new data and resolve errors.

## Getting Started

For detailed instructions and examples, visit the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/) pages.

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

### LLM based text-file evaluation

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

## Community and Support

*   [Documentation](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## DataChain Studio Platform

`DataChain Studio` is a proprietary solution for teams, offering:

*   **Centralized dataset registry** to manage data, code, and dependencies.
*   **Data Lineage** for data sources and derived datasets.
*   **UI for Multimodal Data** (images, videos, PDFs).
*   **Scalable Compute** for large datasets (100M+ files) and in-house AI model inference.
*   **Access control** including SSO and team-based collaboration.