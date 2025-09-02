# DataChain: Transform and Analyze Unstructured Data with Ease

**DataChain is a Python-based AI-data warehouse that simplifies the transformation, analysis, and versioning of unstructured data like images, audio, videos, text, and PDFs.** Learn more about DataChain on [GitHub](https://github.com/iterative/datachain).

## Key Features

*   **Multimodal Data Support:**
    *   Effortlessly version and manage unstructured data, including images, videos, text, PDFs, JSONs, CSVs, and Parquet files.
    *   Works with data stored in S3, GCP, Azure, and local file systems.
    *   Unite files and metadata into versioned, columnar datasets.
*   **Python-Native Processing:**
    *   Seamlessly process and transform data using Python objects and their fields (e.g., float scores, strings, matrices, LLM responses).
    *   Scale to terabyte-sized datasets with built-in parallelization and memory-efficient computing, eliminating the need for SQL or Spark.
*   **Advanced Data Enrichment and Analysis:**
    *   Generate metadata using local AI models and LLM APIs.
    *   Filter, join, and group datasets based on metadata. Search by vector embeddings.
    *   Perform high-performance vectorized operations (sum, count, avg, etc.) on Python objects.
    *   Integrate with PyTorch and TensorFlow, or export data back to storage.
*   **Incremental Processing:**
    *   **Delta Processing:** Process only new or changed files.
    *   **Retry Processing:** Automatically reprocess records with errors or missing results
    *   **Combined Approach:** Process new data and fix errors in a single pipeline

## Use Cases

*   **ETL:** Pythonic framework for describing and running unstructured data transformations.
*   **Analytics:** DataChain datasets provide a table-like API for scalable analytics.
*   **Versioning:** Version unstructured data without moving or creating copies.
*   **Incremental Processing:** Process only new or changed files/records, retry processing records with errors.

## Getting Started

*   Visit the [Quick Start](https://docs.datachain.ai/quick-start) to begin.
*   Explore the full [Documentation](https://docs.datachain.ai/) for in-depth information.

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

## Contribute

Contributions are very welcome. To learn more, see the [Contributor Guide](https://docs.datachain.ai/contributing).