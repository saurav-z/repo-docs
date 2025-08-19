<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**Unlock the power of Large Language Models (LLMs) to transform raw text into structured, actionable insights with LangExtract, a versatile Python library.**  [See the original repo](https://github.com/google/langextract).

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

**Key Features:**

*   **Precise Source Grounding:**  Pinpoint extractions to their exact location in the original text for easy verification.
*   **Reliable Structured Outputs:** Leverage few-shot examples for consistent schema adherence across supported models.
*   **Optimized for Long Documents:**  Efficiently process large texts using chunking, parallelization, and multi-pass extraction.
*   **Interactive Visualization:**  Generate interactive HTML files to review thousands of extracted entities in context.
*   **Flexible LLM Support:** Works with cloud-based LLMs like Gemini, as well as local open-source LLMs via Ollama.
*   **Domain Agnostic:** Define extraction tasks for any domain using simple examples; no fine-tuning needed.
*   **Leverages LLM World Knowledge:**  Control how the LLM uses its world knowledge through prompts and examples.

## Table of Contents

-   [Introduction](#introduction)
-   [Quick Start](#quick-start)
    -   [1. Define Your Extraction Task](#1-define-your-extraction-task)
    -   [2. Run the Extraction](#2-run-the-extraction)
    -   [3. Visualize the Results](#3-visualize-the-results)
    -   [Scaling to Longer Documents](#scaling-to-longer-documents)
-   [Installation](#installation)
    -   [From PyPI](#from-pypi)
    -   [From Source](#from-source)
    -   [Docker](#docker)
-   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
    -   [API Key Sources](#api-key-sources)
    -   [Setting up API key in your environment](#setting-up-api-key-in-your-environment)
-   [Adding Custom Model Providers](#adding-custom-model-providers)
-   [Using OpenAI Models](#using-openai-models)
-   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
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

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents.  It leverages the power of LLMs to identify and organize key details based on user-defined instructions, making it ideal for processing clinical notes, reports, and other text-based data. The extracted data is precisely mapped to its original source, ensuring traceability and ease of verification.

## Quick Start

Get started extracting data in just a few steps:

> **Note:** Using cloud-hosted models (like Gemini) requires setting up an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section.

### 1. Define Your Extraction Task

Craft a prompt that precisely outlines your desired extraction, providing an example to guide the model.

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

Use the `lx.extract` function with your text, prompt, and examples.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default. For more complex tasks, use `gemini-2.5-pro`.  For production, consider Tier 2 Gemini quotas. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Stay informed about Gemini model lifecycles by consulting the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions).

### 3. Visualize the Results

Save the extractions to a `.jsonl` file and generate an interactive HTML visualization for review.

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

> **Note on LLM Knowledge Utilization:** The extraction's focus on text evidence can be adjusted. Modify the task to use the LLM's world knowledge by generating attributes (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`). The level of inference is controlled by your prompt and example attributes.

### Scaling to Longer Documents

Process large texts directly from URLs with parallel processing:

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

The interactive visualization easily handles large result sets. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users. For isolated environments, consider using a virtual environment:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses modern Python packaging with `pyproject.toml` for dependency management:

*Installing with `-e` puts the package in development mode, allowing you to modify the code without reinstalling.*

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

Cloud-hosted models (Gemini, OpenAI) require API keys. On-device models don't.

### API Key Sources

*   [AI Studio](https://aistudio.google.com/app/apikey) for Gemini
*   [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) for enterprise use
*   [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI

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

Extend LangExtract with custom LLM providers using a lightweight plugin system. See the [Provider System Documentation](langextract/providers/README.md) to learn how to:

-   Register a provider with `@registry.register(...)`
-   Publish an entry point for discovery
-   Optionally provide a schema with `get_schema_class()` for structured output
-   Integrate with the factory via `create_model(...)`

## Using OpenAI Models

Requires optional dependency: `pip install langextract[openai]`:

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

For detailed installation, Docker setup, and examples, see [`examples/ollama/`](examples/ollama/).

## More Examples

### *Romeo and Juliet* Full Text Extraction

Process complete documents directly from URLs. This example shows parallel processing and performance optimization for long document processing.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes only. It is not a finished product, is not intended to diagnose or suggest treatment, and should not be used for medical advice.

Examples demonstrating entity and relationship extraction for healthcare applications.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore a live demo of LangExtract structuring radiology reports on HuggingFace Spaces.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines. You must sign a
[Contributor License Agreement](https://cla.developers.google.com/about).

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

Or reproduce the full CI matrix locally with tox:

```bash
tox
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, use is also subject to the
[Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**