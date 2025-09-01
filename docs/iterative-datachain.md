# DataChain: The Python-Powered AI Data Warehouse for Unstructured Data

**DataChain is a powerful Python library for transforming, analyzing, and versioning unstructured data at scale.** ([View the original repo](https://github.com/iterative/datachain))

## Key Features

*   **🔄 Data Versioning:** Manage and version your unstructured data (images, video, text, PDFs, and more) without data duplication.  Works with data stored in S3, GCP, Azure, and local file systems.
*   **🐍 Python-First:** Designed for Python users, allowing you to work with Python objects and leverage built-in parallelization for high-performance processing.
*   **🧠 AI-Ready:**  Enrich data with metadata using local AI models and LLM APIs, including filtering, joining, grouping, and vector search capabilities.
*   **💾 Scalable ETL & Analytics:** Build ETL pipelines and perform analytics on large, unstructured datasets with a dataframe-like API and vectorized engine.
*   **⚡️ Incremental Processing:** Utilize delta processing and automatic retry features to process only new or changed files and handle errors efficiently.

## Use Cases

*   **ETL:** Build Pythonic pipelines for transforming and enriching unstructured data with models, including LLMs.
*   **Analytics:** Perform large-scale analytics on multimodal datasets.
*   **Versioning:** Version data without moving or copying, suitable for large cloud storage.
*   **Incremental Processing:** Process new data and fix errors with delta and retry features.

## Getting Started

Install DataChain:

```bash
pip install datachain
```

Explore the [Quick Start](https://docs.datachain.ai/quick-start) and comprehensive [Docs](https://docs.datachain.ai/) to begin using DataChain.

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

*   [Docs](https://docs.datachain.ai/)
*   [File an issue](https://github.com/iterative/datachain/issues)
*   [Discord Chat](https://dvc.org/chat)
*   [Email](mailto:support@dvc.org)
*   [Twitter](https://twitter.com/DVCorg)

## Contributing

Contributions are welcome! Please see the [Contributor Guide](https://docs.datachain.ai/contributing).

## DataChain Studio Platform

For teams, consider [DataChain Studio](https://studio.datachain.ai/) for:

*   Centralized dataset registry
*   Data lineage
*   UI for multimodal data
*   Scalable compute
*   Access control