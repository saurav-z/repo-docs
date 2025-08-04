<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Information from Unstructured Text

**LangExtract is a powerful Python library that leverages the power of Large Language Models (LLMs) to automatically extract key information from text, transforming unstructured data into structured insights.** ([See the original repo](https://github.com/google/langextract))

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise and Grounded Extractions:** Every extracted piece of information is linked back to its exact source in the original text.
*   **Consistent, Structured Outputs:**  Define your desired output format using examples, ensuring reliable and predictable results, even with complex data.
*   **Optimized for Large Documents:** Efficiently handles lengthy documents through chunking, parallel processing, and multiple extraction passes for higher accuracy.
*   **Interactive Visualization:** Generate dynamic HTML files to easily review and understand extracted entities within their original context.
*   **Flexible LLM Support:** Compatible with a variety of models, including cloud-based (Gemini, OpenAI) and local open-source models (Ollama).
*   **Domain Agnostic:**  Adaptable to any domain; define extraction tasks with just a few examples, without model fine-tuning.
*   **Leverages LLM Knowledge:** Utilizes precise prompting and examples to tap into the LLM's world knowledge for more comprehensive extractions.  *The accuracy of inferred information and its adherence to task specifications depend on the chosen LLM, complexity, prompt instructions, and example design.*

## Getting Started

### 1. Define Your Extraction Task

Create a clear prompt describing your desired extractions, and provide a high-quality example to guide the model.

```python
import langextract as lx
import textwrap

prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

examples = [
    lx.data.ExampleData(
        text="ROMEO. But soft! What light through yonder window breaks? It is the east, and Juliet is the sun.",
        extractions=[
            lx.data.Extraction(
                extraction_class="character",
                extraction_text="ROMEO",
                attributes={"emotional_state": "wonder"}
            ),
            lx.data.Extraction(
                extraction_class="emotion",
                extraction_text="But soft!",
                attributes={"feeling": "gentle awe"}
            ),
            lx.data.Extraction(
                extraction_class="relationship",
                extraction_text="Juliet is the sun",
                attributes={"type": "metaphor"}
            ),
        ]
    )
]
```

### 2. Run the Extraction

Use the `lx.extract` function to process your text.

```python
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**:  `gemini-2.5-flash` offers an excellent balance of speed, cost, and quality and is the recommended default.  For more complex tasks, consider `gemini-2.5-pro`.  For production, consider a Tier 2 Gemini quota to avoid rate limits. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Stay informed about the latest Gemini versions by consulting the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions).

### 3. Visualize the Results

Save your extractions to a `.jsonl` file and create an interactive HTML visualization.

```python
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content)
```

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

### Scaling to Longer Documents

Process entire documents directly from URLs with parallel processing and enhanced sensitivity:

```python
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,    # Improves recall through multiple passes
    max_workers=20,         # Parallel processing for speed
    max_char_buffer=1000    # Smaller contexts for better accuracy
)
```

**[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

## Installation

### From PyPI

```bash
pip install langextract
```

### From Source

```bash
git clone https://github.com/google/langextract.git
cd langextract
pip install -e .
```

## API Key Setup

API keys are required for cloud-hosted LLMs like Gemini and OpenAI.

### API Key Sources

*   [AI Studio](https://aistudio.google.com/app/apikey) for Gemini models
*   [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) for enterprise use
*   [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI models

### Setting up API key in your environment

**Option 1: Environment Variable**

```bash
export LANGEXTRACT_API_KEY="your-api-key-here"
```

**Option 2: .env File (Recommended)**

Add your API key to a `.env` file:

```bash
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
EOF

echo '.env' >> .gitignore
```

In your Python code:
```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash"
)
```

**Option 3: Direct API Key (Not Recommended for Production)**

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"
)
```

## Using OpenAI Models

```python
from langextract.inference import OpenAILanguageModel

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    language_model_type=OpenAILanguageModel,
    model_id="gpt-4o",
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False`.

## More Examples

*   ***Romeo and Juliet* Full Text Extraction:**  Process the entire play directly from a URL. **[View Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**
*   **Medication Extraction:** Extract structured medical information from clinical text.  **[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**
*   **Radiology Report Structuring: RadExtract:**  A live demo on HuggingFace Spaces. **[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for details.

## Testing

```bash
pip install -e ".[test]"
pytest tests
```

or use `tox`.

## Development

```bash
./autoformat.sh
isort langextract tests --profile google --line-length 80
pyink langextract tests --config pyproject.toml

pre-commit install
pre-commit run --all-files
pylint --rcfile=.pylintrc langextract tests
```

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, use is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Extract your insights today!**