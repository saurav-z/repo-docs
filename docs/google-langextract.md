<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Unlock Structured Data from Unstructured Text with LLMs

**Effortlessly extract key information from text using Large Language Models (LLMs) with LangExtract, a powerful Python library.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   ✅ **Precise Source Grounding:** Highlight extracted data directly in the source text for easy verification.
*   ✅ **Reliable Structured Outputs:**  Enforce a consistent output schema using few-shot examples, ensuring robust and structured results.
*   ✅ **Optimized for Long Documents:**  Handles large documents with efficient text chunking, parallel processing, and multi-pass extraction.
*   ✅ **Interactive Visualization:** Generate interactive HTML files to visualize and explore extracted entities in context.
*   ✅ **Flexible LLM Support:** Works with cloud-based models (Gemini, OpenAI) and local open-source LLMs (Ollama).
*   ✅ **Domain Agnostic:** Define extractions for any domain with just a few examples, without model fine-tuning.
*   ✅ **Leverages LLM World Knowledge:** Utilize precise prompt wording and few-shot examples to influence how the extraction task may utilize LLM knowledge.

## Table of Contents

-   [Introduction](#introduction)
-   [Why LangExtract?](#why-langextract)
-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
-   [Adding Custom Model Providers](#adding-custom-model-providers)
-   [Using OpenAI Models](#using-openai-models)
-   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
-   [More Examples](#more-examples)
    -   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    -   [Medication Extraction](#medication-extraction)
    -   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
-   [Community Providers](#community-providers)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
-   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents using the power of Large Language Models (LLMs). It's perfect for processing clinical notes, reports, and other textual data, enabling you to identify and organize key details while maintaining traceability to the source text.

## Why LangExtract?

*   **Precise Source Grounding:** Maps every extraction to its exact location in the source text.
*   **Reliable Structured Outputs:** Enforces a consistent output schema based on few-shot examples.
*   **Optimized for Long Documents:** Efficiently handles large document extraction.
*   **Interactive Visualization:** Generates self-contained HTML files for interactive review.
*   **Flexible LLM Support:** Supports cloud-based and local LLMs.
*   **Adaptable to Any Domain:** Easily define extraction tasks for any domain.
*   **Leverages LLM World Knowledge:** Harnesses LLM knowledge through prompt wording and examples.

## Quick Start

> **Note:** Cloud-hosted models require an API key.  See the [API Key Setup](#api-key-setup-for-cloud-models) section.

Get started extracting structured information with a few lines of code.

### 1. Define Your Extraction Task

Create a prompt describing what to extract, along with a high-quality example.

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

Provide your input text and the prompt to the `lx.extract` function.

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

> **Model Selection**: `gemini-2.5-flash` is recommended for speed, cost, and quality. For complex tasks, use `gemini-2.5-pro`.  For production, consider Tier 2 Gemini quota (see [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2)).
>
> **Model Lifecycle**: Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest versions.

### 3. Visualize the Results

Save extractions to a `.jsonl` file and generate an interactive HTML visualization.

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

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example extracts directly from the text.  Modify the task to utilize the LLM's world knowledge.

### Scaling to Longer Documents

Process entire documents directly from URLs with parallel processing:

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

**[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results and performance insights.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users. Use a virtual environment for isolation:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

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

Set up an API key when using cloud-hosted models (Gemini or OpenAI). On-device models (e.g., Ollama) do not require an API key.

### API Key Sources

*   [AI Studio](https://aistudio.google.com/app/apikey) for Gemini models
*   [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) for enterprise
*   [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI models

### Setting up API key in your environment

**Option 1: Environment Variable**

```bash
export LANGEXTRACT_API_KEY="your-api-key-here"
```

**Option 2: .env File (Recommended)**

Add your API key to a `.env` file:

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

## Adding Custom Model Providers

Extend LangExtract with custom LLM providers using a lightweight plugin system.

*   Add new model support independently.
*   Distribute your provider as a separate package.
*   Keep custom dependencies isolated.
*   Override or extend built-in providers.

See [Provider System Documentation](langextract/providers/README.md).

## Using OpenAI Models

Requires the optional dependency: `pip install langextract[openai]`

```python
import langextract as lx
import os

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

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False`.

## Using Local LLMs with Ollama

Requires Ollama.

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

See [`examples/ollama/`](examples/ollama/) for details.

## More Examples

Explore these examples demonstrating LangExtract's capabilities.

### *Romeo and Juliet* Full Text Extraction

Extract information from the full text of *Romeo and Juliet* from Project Gutenberg.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This is for illustrative purposes only and not for medical advice.

Extract structured medical information from clinical text.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Interactive demo on HuggingFace Spaces.  Try RadExtract directly in your browser.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with community-created plugins.

**[View Community Provider Plugins](COMMUNITY_PROVIDERS.md)**

For plugin creation instructions, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md). You must sign a [Contributor License Agreement](https://cla.developers.google.com/about).

## Testing

```bash
# Clone the repository
git clone https://github.com/google/langextract.git
cd langextract

# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests
```

Or reproduce the CI matrix locally with tox:

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

See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for full development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and, for health-related applications, the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**