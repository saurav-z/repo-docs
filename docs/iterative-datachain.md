# DataChain: Transform, Analyze, and Version Your Unstructured Data

**DataChain empowers you to efficiently process and analyze unstructured data like images, videos, and text, all within a Pythonic framework.**  [See the original repo](https://github.com/iterative/datachain)

## Key Features

*   **Unstructured Data Warehousing:** Efficiently process and analyze images, audio, videos, text, PDFs, and other unstructured data types.
*   **Version Control for Data:** Version unstructured data without data duplication, using references to cloud storage like S3, GCP, Azure, and local file systems.
*   **Python-Native:** Leverage Python objects and run Python code on large datasets with built-in parallelization.
*   **Data Enrichment & Processing:** Enrich and process data with LLMs, filter, join, and group datasets by metadata. Perform high-performance vectorized operations on Python objects and integrate with PyTorch and TensorFlow.
*   **ETL Capabilities:**  Transform and enrich data using a Pythonic framework, perfect for applying models, including LLMs.
*   **Delta & Retry Processing:** Process only new or changed files and automatically retry failed processes for efficient workflows.
*   **Analytics Engine:** DataChain datasets combine all data object information in a single place, offering a DataFrame-like API and a vectorized engine for scalable analytics.

## Getting Started

Install DataChain using pip:

```bash
pip install datachain
```

Explore the documentation and quick start guides to begin using DataChain:

*   [Quick Start](https://docs.datachain.ai/quick-start)
*   [Docs](https://docs.datachain.ai/)

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

### LLM-based Text File Evaluation

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

## DataChain Studio Platform

[DataChain Studio](https://studio.datachain.ai/) offers a proprietary solution for teams with:

*   Centralized dataset registry
*   Data Lineage
*   UI for Multimodal Data
*   Scalable Compute
*   Access control