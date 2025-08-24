# DataChain: AI-Powered Data Warehouse for Unstructured Data

**DataChain is a Python-based data warehouse that transforms and analyzes unstructured data like images, audio, and text, streamlining your AI and data workflows.** ([View the original repo](https://github.com/iterative/datachain))

## Key Features

*   **Unstructured Data Management:** Version and manage images, videos, text, PDFs, and more, without data duplication, directly from cloud storage (S3, GCP, Azure) or local file systems.
*   **Pythonic Data Transformations:** Utilize a Python-friendly framework for data transformations, enrichment, and model application, including large language models (LLMs).
*   **Efficient Data Analytics:** Perform scalable data analysis on combined files and metadata within a dataframe-like API with a vectorized engine.
*   **Incremental Processing:** Leverage delta processing and retry features for efficient workflows.
    *   **Delta Processing:** Process only new or changed files.
    *   **Retry Processing:** Automatically reprocess records with errors.
    *   **Combined Approach:** Process new data and fix errors in a single pipeline.
*   **Data Enrichment and Processing:** Generate metadata with AI models and LLM APIs.
*   **Metadata Management:** Filter, join, and group datasets by metadata, and utilize vector embeddings for advanced search.

## Use Cases

*   **ETL (Extract, Transform, Load):** Transform unstructured data.
*   **Analytics:** Analyze data at scale.
*   **Versioning:** Manage data versions without data duplication.
*   **Incremental Processing:** Efficiently handle changing datasets.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

For detailed instructions and examples, see the [Quick Start](https://docs.datachain.ai/quick-start) and [Docs](https://docs.datachain.ai/).

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

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## Contributing

Contributions are welcome. See the [Contributor Guide](https://docs.datachain.ai/contributing).

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) is a proprietary solution that offers:

*   Centralized dataset registry.
*   Data lineage.
*   UI for multimodal data.
*   Scalable compute.
*   Access control.