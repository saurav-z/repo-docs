<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**LangExtract is a powerful Python library leveraging Large Language Models (LLMs) to automatically extract structured information from unstructured text documents.** ([View the original repository](https://github.com/google/langextract))

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Source Grounding:** Extracts are linked to their exact location in the source text for easy verification.
*   **Reliable Structured Outputs:** Consistent output schemas are enforced based on your few-shot examples, ensuring dependable results.
*   **Optimized for Long Documents:** Efficiently handles large documents using chunking, parallel processing, and multiple extraction passes.
*   **Interactive Visualization:** Generates self-contained, interactive HTML visualizations for easy review of extracted entities.
*   **Flexible LLM Support:** Works with a variety of LLMs, including cloud-based (Gemini, OpenAI) and local open-source models (Ollama).
*   **Domain Agnostic:** Define extraction tasks for any domain using a few examples; no model fine-tuning needed.
*   **Leverages LLM World Knowledge:** Utilizes precise prompting and few-shot examples to guide LLMs in leveraging their knowledge.

## Table of Contents

-   [Quick Start](#quick-start)
-   [Installation](#installation)
    -   [From PyPI](#from-pypi)
    -   [From Source](#from-source)
    -   [Docker](#docker)
-   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
-   [More Examples](#more-examples)
    -   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    -   [Medication Extraction](#medication-extraction)
    -   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
    -   [Code Formatting](#code-formatting)
    -   [Pre-commit Hooks](#pre-commit-hooks)
    -   [Linting](#linting)
-   [Disclaimer](#disclaimer)

## Quick Start

Quickly extract structured information by defining your extraction task, providing examples, and running the `lx.extract` function.

### 1. Define Your Extraction Task

Create a prompt that describes what you want to extract and guide the model with a high-quality example.

```python
import langextract as lx
import textwrap

# 1. Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# 2. Provide a high-quality example to guide the model
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

Provide your input text, prompt materials, and choose your preferred LLM.

```python
# The input text to be processed
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**:  `gemini-2.5-flash` is the recommended default. For complex tasks use `gemini-2.5-pro`.  For large-scale use, a Tier 2 Gemini quota is suggested. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for model updates.

### 3. Visualize the Results

Save extractions to a `.jsonl` file, then generate an interactive HTML visualization for easy review.

```python
# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content)
```

This produces an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to the text. The task could be modified to generate attributes that draw more heavily from the LLM's world knowledge. The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

Process full documents from URLs with parallel processing and enhanced sensitivity:

```python
# Process Romeo & Juliet directly from Project Gutenberg
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

This approach extracts hundreds of entities while maintaining high accuracy. The interactive visualization makes it easy to explore large result sets. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results.

## Installation

### From PyPI

```bash
pip install langextract
```

*For isolated environments:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses modern Python packaging with `pyproject.toml`:

*Installing with `-e` puts the package in development mode:*

```bash
git clone https://github.com/google/langextract.git
cd langextract

# For basic installation:
pip install -e .

# For development (includes linting tools):
pip install -e ".[dev]"

# For testing (includes pytest):
pip install -e ".[test]"
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

Requires an API key for cloud-hosted models (Gemini, OpenAI). Local models don't require an API key.

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

```bash
# Add API key to .env file
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
EOF

# Keep your API key secure
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
    api_key="your-api-key-here"  # Only use this for testing/development
)
```

## Using OpenAI Models

Example OpenAI configuration:

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

### *Romeo and Juliet* Full Text Extraction

Extract from the full text of *Romeo and Juliet* (147,843 characters) using parallel processing and performance optimizations.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes of LangExtract's baseline capability only. It does not represent a finished or approved product, is not intended to diagnose or suggest treatment of any disease or condition, and should not be used for medical advice.

Extract structured medical information from clinical text, including entity and relationship extractions.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live interactive demo on HuggingFace Spaces.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome!  See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md). You must sign a
[Contributor License Agreement](https://cla.developers.google.com/about).

## Testing

To run tests locally:

```bash
# Clone the repository
git clone https://github.com/google/langextract.git
cd langextract

# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests
```

Or reproduce the full CI matrix locally with tox:

```bash
tox  # runs pylint + pytest on Python 3.10 and 3.11
```

## Development

### Code Formatting

```bash
# Auto-format all code
./autoformat.sh

# Or run formatters separately
isort langextract tests --profile google --line-length 80
pyink langextract tests --config pyproject.toml
```

### Pre-commit Hooks

```bash
pre-commit install  # One-time setup
pre-commit run --all-files  # Manual run
```

### Linting

```bash
pylint --rcfile=.pylintrc langextract tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and, for health-related applications, the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**