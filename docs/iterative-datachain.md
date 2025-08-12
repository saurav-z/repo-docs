# DataChain: Transform, Analyze, and Version Your Unstructured Data

**DataChain is a Python-based AI-data warehouse that simplifies the management and analysis of unstructured data, enabling efficient processing and versioning.** ([Original Repository](https://github.com/iterative/datachain))

## Key Features

*   **Seamless Data Versioning:** Version your unstructured data without data duplication. Supports various storage backends like S3, GCP, Azure, and local file systems.
*   **Multimodal Data Support:** Works with various data types, including images, videos, text, PDFs, JSONs, CSVs, and parquet files.
*   **Python-Native:** Designed for Python users, providing a familiar and efficient environment for data manipulation.
*   **Efficient Data Enrichment & Processing:**
    *   Integrate with local AI models and LLM APIs.
    *   Filter, join, and group datasets by metadata. Search by vector embeddings.
    *   Perform high-performance vectorized operations.
    *   Seamlessly integrate with PyTorch and TensorFlow.
*   **Incremental Processing:**
    *   Process only new or changed files.
    *   Automatically retry records with errors.
    *   Combine new data processing and error correction in a single pipeline.

## Use Cases

*   **ETL:** Pythonic framework for transforming and enriching unstructured data.
*   **Analytics:** Analyze datasets at scale with a dataframe-like API and vectorized engine.
*   **Versioning:** Efficiently version large datasets without data duplication.
*   **Incremental Processing:** Optimize workflows with delta and retry features.

## Getting Started

For detailed instructions and examples, please refer to the following resources:

*   [Quick Start](https://docs.datachain.ai/quick-start)
*   [Documentation](https://docs.datachain.ai/)

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