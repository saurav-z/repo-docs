<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Extract Structured Data from Unstructured Text with LLMs

**Effortlessly transform unstructured text into structured data using the power of Large Language Models with LangExtract.** ([Original Repo](https://github.com/google/langextract))

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Source Grounding:** Trace extractions back to the source text with highlighting.
*   **Reliable Structured Outputs:** Get consistent, structured results based on your examples.
*   **Optimized for Long Documents:** Efficiently processes large documents with chunking and parallel processing.
*   **Interactive Visualization:** Instantly visualize and review extractions in context with an HTML file.
*   **Flexible LLM Support:** Works with cloud-based (Gemini, OpenAI) and local LLMs (Ollama).
*   **Domain Agnostic:** Define extraction tasks for any domain with just a few examples.
*   **Leverages LLM Knowledge:** Fine-tune extractions with prompt engineering and example attributes.

## Table of Contents

-   [Introduction](#introduction)
-   [Why LangExtract?](#why-langextract)
-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup](#api-key-setup-for-cloud-models)
-   [Adding Custom Model Providers](#adding-custom-model-providers)
-   [Using OpenAI Models](#using-openai-models)
-   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
-   [Examples](#more-examples)
    -   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    -   [Medication Extraction](#medication-extraction)
    -   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
-   [Community Providers](#community-providers)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
-   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a powerful Python library that utilizes Large Language Models (LLMs) to extract structured information from unstructured text. This makes it ideal for processing a wide range of documents, such as clinical notes, reports, and more. By defining your extraction requirements with simple instructions and examples, LangExtract identifies and organizes key details while maintaining source text correspondence.

## Why LangExtract?

LangExtract offers a robust and flexible solution for information extraction, providing:

1.  **Precise Source Grounding:** Accurate extraction linked to source text locations.
2.  **Reliable Structured Outputs:** Consistent output schemas based on your examples and LLM capabilities.
3.  **Optimized for Long Documents:** Handles the "needle-in-a-haystack" challenge with efficient processing.
4.  **Interactive Visualization:** Generate self-contained, interactive HTML files for immediate review.
5.  **Flexible LLM Support:** Use your preferred models, from cloud-based LLMs to local open-source models.
6.  **Adaptable to Any Domain:** Define custom extraction tasks with a few examples.
7.  **Leverages LLM World Knowledge:** Customize extractions by strategically utilizing LLM world knowledge through prompting.

## Quick Start

> **Note:** Cloud models require an API key. See [API Key Setup](#api-key-setup-for-cloud-models) for details.

Get started extracting structured information with just a few lines of code.

### 1. Define Your Extraction Task

Create a prompt that describes what you want to extract and provide an example.

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

Provide your input text, the prompt, and the example data to the `lx.extract` function.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default, offering a good balance of speed, cost, and quality. For complex tasks, `gemini-2.5-pro` might be preferred. For production use, consider a Tier 2 Gemini quota for better throughput. Refer to [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for more information.
>
> **Model Lifecycle**: Check the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest model versions.

### 3. Visualize the Results

Save extractions to a `.jsonl` file, a common format for language model data.  LangExtract can generate an interactive HTML visualization from the file.

```python
# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

This creates an animated and interactive HTML file like the example below:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example extracts information directly from the text. The task can be modified to draw more heavily from the LLM's world knowledge (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`). The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

Process entire documents using URLs with parallel processing and enhanced sensitivity:

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

This approach can accurately extract from full novels. The interactive visualization can easily handle large result sets. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for details.

## Installation

### From PyPI

```bash
pip install langextract
```

*For isolated environments, consider a virtual environment:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses modern Python packaging with `pyproject.toml`.

*Installing with `-e` puts the package in development mode.*

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

Cloud-hosted models (Gemini, OpenAI) require an API key. On-device models do not.

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

Create a `.env` file:

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

For testing only:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only use this for testing/development
)
```

**Option 4: Vertex AI (Service Accounts)**

Use [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) for authentication with service accounts:

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    language_model_params={
        "vertexai": True,
        "project": "your-project-id",
        "location": "global"  # or regional endpoint
    }
)
```

## Adding Custom Model Providers

LangExtract's plugin system lets you add support for new LLMs easily.

- Add support without changing core code
- Distribute your provider as a separate package
- Keep custom dependencies isolated
- Override/extend built-in providers via priority

See the [Provider System Documentation](langextract/providers/README.md) for details.

## Using OpenAI Models

LangExtract supports OpenAI models (requires `pip install langextract[openai]`):

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",  # Automatically selects OpenAI provider
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

*   OpenAI models require `fence_output=True` and `use_schema_constraints=False`.

## Using Local LLMs with Ollama

LangExtract supports local inference using Ollama, without needing API keys:

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",  # Automatically selects Ollama provider
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False
)
```

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

For more details, see [`examples/ollama/`](examples/ollama/).

## Examples

### *Romeo and Juliet* Full Text Extraction

Processes complete documents directly from URLs. This example showcases parallel processing, sequential extraction passes, and optimizations for long document processing.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This is for illustrative purposes only and not intended for medical advice.

Extract structured medical information from clinical text. This demonstrates entity recognition and relationship extraction for healthcare applications.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

A live demo on HuggingFace Spaces showing how LangExtract can automatically structure radiology reports.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with custom model providers! Check out our [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry or create your own using the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines. You must sign a [Contributor License Agreement](https://cla.developers.google.com/about).

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

### Ollama Integration Testing

```bash
# Test Ollama integration (requires Ollama running with gemma2:2b model)
tox -e ollama-integration
```

## Development

### Code Formatting

This project uses automated formatting:

```bash
# Auto-format all code
./autoformat.sh

# Or run formatters separately
isort langextract tests --profile google --line-length 80
pyink langextract tests --config pyproject.toml
```

### Pre-commit Hooks

For automatic formatting checks:

```bash
pre-commit install  # One-time setup
pre-commit run --all-files  # Manual run
```

### Linting

Run linting before submitting PRs:

```bash
pylint --rcfile=.pylintrc langextract tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Disclaimer

This is not an officially supported Google product.  Please cite and acknowledge usage appropriately. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**