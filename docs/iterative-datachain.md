# DataChain: Transform and Analyze Unstructured Data with Python

**DataChain** is a Python-based framework designed to efficiently transform, analyze, and version unstructured data like images, audio, video, and text directly from your storage, all without data duplication.  ([View on GitHub](https://github.com/iterative/datachain))

## Key Features:

*   **Efficient Data Warehousing:**
    *   Manages unstructured data (images, videos, text, PDFs, and more) and associated metadata.
    *   Integrates with external storage (S3, GCP, Azure, local file systems) for direct processing.
    *   Avoids data duplication by storing references to your data.
    *   Unites files and metadata into persistent, versioned, columnar datasets.
*   **Python-Native Interface:**
    *   Provides a Pythonic API for data manipulation.
    *   Works with Python objects and object fields: floats, strings, matrices, and LLM responses.
    *   Enables high-scale operations on large datasets (terabytes) with built-in parallelization.
    *   No SQL or Spark knowledge needed.
*   **Powerful Data Processing:**
    *   ETL (Extract, Transform, Load) framework for building data pipelines.
    *   Allows for enrichment with local AI models and LLM APIs.
    *   Offers dataframe-like API with vectorized engine for large-scale analytics.
    *   Supports filtering, joining, and grouping data based on metadata.
    *   Enables high-performance vectorized operations: sum, count, average, etc.
    *   Allows integration with PyTorch and TensorFlow.
*   **Advanced Versioning and Incremental Processing:**
    *   Supports versioning without copying data.
    *   Offers delta processing to process only new or changed files.
    *   Includes retry mechanisms for handling errors and ensuring data integrity.

## Use Cases:

*   **ETL Pipelines:** Build robust data transformation pipelines.
*   **Data Analytics:** Analyze large, unstructured datasets.
*   **Data Versioning:** Manage and track changes to datasets efficiently.
*   **Incremental Processing:** Process new data and handle errors effectively.

## Getting Started

See the [Quick Start](https://docs.datachain.ai/quick-start) and comprehensive [Docs](https://docs.datachain.ai/) to learn more.

```bash
pip install datachain
```

## Examples:

**Download Subset of Files Based on Metadata**

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

**Incremental Processing with Error Handling**

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

**LLM based text-file evaluation**

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

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)