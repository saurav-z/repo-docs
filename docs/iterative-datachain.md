# DataChain: Transform and Analyze Unstructured Data at Scale

**DataChain** is a powerful Python-based data warehouse designed for efficient transformation and analysis of unstructured data like images, audio, video, text, and PDFs. [Visit the original repository](https://github.com/iterative/datachain)

[![PyPI](https://img.shields.io/pypi/v/datachain.svg)](https://pypi.org/project/datachain/)
[![Python Version](https://img.shields.io/pypi/pyversions/datachain)](https://pypi.org/project/datachain)
[![Codecov](https://codecov.io/gh/iterative/datachain/graph/badge.svg?token=byliXGGyGB)](https://codecov.io/gh/iterative/datachain)
[![Tests](https://github.com/iterative/datachain/actions/workflows/tests.yml/badge.svg)](https://github.com/iterative/datachain/actions/workflows/tests.yml)
[![DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/iterative/datachain)

## Key Features

*   **Multimodal Data Support:** Process and version various data types (images, video, text, PDFs, JSONs, CSVs, parquet, etc.) without data duplication.
*   **Python-Friendly:** Seamlessly integrate with your existing Python workflows. Utilize Python objects, operate on object fields and run high-scale calculations.
*   **Data Enrichment and Processing:** Enrich your data with metadata, apply local AI models and LLM APIs, and perform efficient filtering, joining, and grouping operations.
*   **Efficient Versioning:** Version unstructured data without moving or creating data copies, supporting references to S3, GCP, Azure, and local file systems.
*   **Incremental Processing:** Utilize delta and retry features for efficient processing of new or changed data and handling errors.

## Use Cases

1.  **ETL (Extract, Transform, Load):** Build Pythonic pipelines for transforming and enriching unstructured data, including applying models like LLMs.
2.  **Analytics:** Analyze large-scale datasets with a dataframe-like API and vectorized engine.
3.  **Versioning:** Efficiently manage and version data stored in cloud storage without data duplication (ideal for large datasets of images, videos, or PDFs).
4.  **Incremental Processing:** Process only new or changed files, automatically retry failed processes.

## Getting Started

1.  **Installation:**

    ```bash
    pip install datachain
    ```

2.  **Quick Start:**
    Visit the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/) for detailed guides and examples.

## Example Usage

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

## Contributing

Contributions are very welcome. See the [Contributor Guide](https://docs.datachain.ai/contributing) to learn more.