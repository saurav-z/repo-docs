<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**Unlock the power of LLMs to transform unstructured text into structured data with LangExtract, a versatile Python library that simplifies information extraction.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17015089.svg)](https://doi.org/10.5281/zenodo.17015089)

## Key Features

*   **Precise Source Grounding:** Easily trace extractions back to their exact source within the text.
*   **Reliable Structured Outputs:** Enforces consistent data formats using few-shot examples and schema constraints.
*   **Optimized for Long Documents:** Efficiently handles large documents through chunking, parallel processing, and multiple extraction passes.
*   **Interactive Visualization:** Instantly visualize and review extracted entities in their original context with an interactive HTML file.
*   **Flexible LLM Support:** Works with a variety of models, from cloud-based (Gemini, OpenAI) to local open-source LLMs (Ollama).
*   **Domain Agnostic:** Adaptable to any domain – define extraction tasks with just a few examples, no model fine-tuning required.
*   **Leverages LLM Knowledge:** Enhance extraction with precise prompt wording and few-shot examples, utilizing LLM's world knowledge.

## Table of Contents

*   [Introduction](#introduction)
*   [Quick Start](#quick-start)
    *   [1. Define Your Extraction Task](#1-define-your-extraction-task)
    *   [2. Run the Extraction](#2-run-the-extraction)
    *   [3. Visualize the Results](#3-visualize-the-results)
    *   [Scaling to Longer Documents](#scaling-to-longer-documents)
*   [Installation](#installation)
    *   [From PyPI](#from-pypi)
    *   [From Source](#from-source)
    *   [Docker](#docker)
*   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
    *   [API Key Sources](#api-key-sources)
    *   [Setting up API key in your environment](#setting-up-api-key-in-your-environment)
*   [Adding Custom Model Providers](#adding-custom-model-providers)
*   [Using OpenAI Models](#using-openai-models)
*   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
*   [More Examples](#more-examples)
    *   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    *   [Medication Extraction](#medication-extraction)
    *   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
*   [Community Providers](#community-providers)
*   [Contributing](#contributing)
*   [Testing](#testing)
    *   [Ollama Integration Testing](#ollama-integration-testing)
*   [Development](#development)
    *   [Code Formatting](#code-formatting)
    *   [Pre-commit Hooks](#pre-commit-hooks)
    *   [Linting](#linting)
*   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to streamline the extraction of structured information from unstructured text documents using the power of Large Language Models (LLMs). Whether you're working with clinical notes, reports, or any text-based data, LangExtract helps you identify, organize, and extract key details efficiently while maintaining source text integrity.

## Quick Start

> **Note:**  Using cloud-hosted models like Gemini requires an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section for instructions.

Get started with structured information extraction using just a few lines of code.

### 1. Define Your Extraction Task

Create a prompt that describes what you want to extract and provide a high-quality example to guide the model.

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

Provide your input text and the prompt materials to the `lx.extract` function.

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

> **Model Selection**:  `gemini-2.5-flash` is the recommended default, providing a good balance of speed, cost, and quality. For complex tasks, `gemini-2.5-pro` might be preferred. For large-scale production use, Tier 2 Gemini quota is suggested to increase throughput. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**:  Note that Gemini models have a lifecycle with defined retirement dates. Users should consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed about the latest stable and legacy versions.

### 3. Visualize the Results

Save extractions to a `.jsonl` file, then generate an interactive HTML visualization to review the entities.

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

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:**  This example shows extractions directly tied to the text. You can adjust the task to use more LLM knowledge (e.g., adding `"identity": "Capulet family daughter"`) The degree of text-evidence vs. knowledge-inference depends on your prompt and example attributes.

### Scaling to Longer Documents

Process entire documents with parallel processing.

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

This approach extracts hundreds of entities accurately.  The interactive visualization handles large datasets seamlessly.  **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for details and performance.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users. Consider a virtual environment for isolated installs:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses `pyproject.toml` for dependency management:

*Installing with `-e` puts the package in development mode.*

```bash
git clone https://github.com/google/langextract.git
cd langextract

# Basic install:
pip install -e .

# Development install (includes linting tools):
pip install -e ".[dev]"

# Testing install (includes pytest):
pip install -e ".[test]"
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

Cloud-hosted models (Gemini or OpenAI) require an API key. On-device models do not.

### API Key Sources

Get API keys from:

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

Provide the API key directly in your code (not recommended for production):

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only for testing/development
)
```

**Option 4: Vertex AI (Service Accounts)**

Use [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) with service accounts:

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

LangExtract supports custom LLM providers via a plugin system.

*   Add new model support independently.
*   Distribute your provider as a separate package.
*   Keep dependencies isolated.
*   Override built-in providers via priority.

See the [Provider System Documentation](langextract/providers/README.md) for detailed instructions on creating a provider plugin.

## Using OpenAI Models

Requires `pip install langextract[openai]`:

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

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False`.

## Using Local LLMs with Ollama

LangExtract supports local inference using Ollama:

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

For detailed setup, Docker and examples, see [`examples/ollama/`](examples/ollama/).

## More Examples

### *Romeo and Juliet* Full Text Extraction

Process the full text of *Romeo and Juliet* from Project Gutenberg.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:**  This is for illustration only and is not a finished or approved product, nor is it for medical advice.

Demonstrates extraction of medical information from clinical text.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Interactive demo on HuggingFace Spaces for structuring radiology reports.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with custom model providers.  Check out the [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry.  See the [Custom Provider Plugin Example](examples/custom_provider_plugin/) for plugin creation instructions.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to get started.  You must sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

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

Or reproduce the full CI matrix with tox:

```bash
tox
```

### Ollama Integration Testing

Requires Ollama installed locally:

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

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, use is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**