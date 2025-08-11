# DataChain: Transform and Analyze Unstructured Data at Scale

**DataChain is a Python-based AI-data warehouse designed to efficiently process and analyze unstructured data like images, audio, videos, text, and PDFs.**  [Explore the DataChain Repository](https://github.com/iterative/datachain)

## Key Features

*   **Effortless Data Versioning:**
    *   Version unstructured data without data duplication, working directly with data in S3, GCP, Azure, and local file systems.
    *   Supports a wide variety of multimodal data types: images, video, text, PDFs, JSONs, CSVs, and more.
    *   Combines files and metadata into persistent, versioned, columnar datasets.

*   **Pythonic Data Processing:**
    *   Works seamlessly with Python objects and their fields, including float scores, strings, matrixes, and LLM response objects.
    *   Enables high-scale data processing of terabytes-sized datasets with built-in parallelization and memory-efficient computing.
    *   Eliminates the need for SQL or Spark.

*   **Powerful Data Enrichment and Analysis:**
    *   Generate rich metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata. Includes support for vector embeddings.
    *   Perform high-performance vectorized operations on Python objects: sum, count, avg, etc.
    *   Integrates with PyTorch and TensorFlow, allowing seamless integration with existing machine learning workflows and easy data export.

## Use Cases

*   **ETL:** Create Pythonic frameworks for describing and running unstructured data transformations, including applying models to data (e.g., LLMs).
*   **Analytics:** Analyze datasets with a table-like API and a vectorized engine, enabling efficient analytics on large-scale data.
*   **Versioning:** Version and track data without moving or copying files, perfect for handling large datasets like image or video libraries.
*   **Incremental Processing:** Utilize delta and retry features for efficient data processing, including processing only new or changed files, automatically reprocessing records with errors, and combining these approaches.

## Getting Started

For detailed instructions and more information, please see the [Quick Start](https://docs.datachain.ai/quick-start) and the [DataChain Documentation](https://docs.datachain.ai/).

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

### LLM-based Text File Evaluation

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
*   [File an Issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## Contributing

Contributions are very welcome!  See the [Contributor Guide](https://docs.datachain.ai/contributing) to learn more.

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution for teams that offers:

*   Centralized dataset registry
*   Data Lineage
*   UI for Multimodal Data
*   Scalable Compute
*   Access control including SSO and team based collaboration.