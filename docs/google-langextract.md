<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Extract Structured Data from Unstructured Text with LLMs

**Unlock the power of Large Language Models (LLMs) to automatically extract and organize key information from your text documents.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17015089.svg)](https://doi.org/10.5281/zenodo.17015089)

## Key Features

*   **Precise Source Grounding:**  Pinpoints extracted data to its exact location in the original text.
*   **Reliable Structured Outputs:**  Ensures consistent, structured results based on your provided examples and schema.
*   **Optimized for Long Documents:**  Handles large documents efficiently using chunking, parallel processing, and multiple passes.
*   **Interactive Visualization:**  Generates interactive HTML files for easy review and exploration of extracted entities.
*   **Flexible LLM Support:** Works with cloud-based LLMs (Gemini, OpenAI) and local, open-source models via Ollama.
*   **Domain Agnostic:** Define extraction tasks for any domain with just a few examples, without the need for fine-tuning.

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
-   [API Key Setup](#api-key-setup-for-cloud-models)
    -   [API Key Sources](#api-key-sources)
    -   [Setting up API key in your environment](#setting-up-api-key-in-your-environment)
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
    -   [Code Formatting](#code-formatting)
    -   [Pre-commit Hooks](#pre-commit-hooks)
    -   [Linting](#linting)
-   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to simplify the process of extracting structured information from unstructured text using the power of Large Language Models (LLMs). Whether you're working with clinical notes, reports, or other text-based documents, LangExtract enables you to identify and organize key details, ensuring the extracted data is accurately grounded in its source.

## Quick Start

Get started extracting structured data in just a few steps!

### 1. Define Your Extraction Task

Create a prompt and provide a high-quality example to guide the model.

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

Use the `lx.extract` function to process your input text with the prompt and examples.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default, offering a good balance of speed, cost, and quality.  For complex tasks, `gemini-2.5-pro` may give superior results. For production use, consider a Tier 2 Gemini quota for increased throughput and to avoid rate limits. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Note that Gemini models have a lifecycle. Users should consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed about the latest versions.

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

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example focuses on extractions directly tied to the text. The task can be altered to leverage the LLM's world knowledge. (e.g., adding `"identity": "Capulet family daughter"`). The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

Process entire documents with parallel processing:

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

**[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for details and insights.

## Installation

### From PyPI

```bash
pip install langextract
```

*Consider using a virtual environment for isolated environments:*

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

When using LangExtract with cloud-hosted models (like Gemini or OpenAI), you'll need an API key.

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

Extend LangExtract with custom LLM providers.  See the [Provider System Documentation](langextract/providers/README.md).

## Using OpenAI Models

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

For details, see [`examples/ollama/`](examples/ollama/).

## More Examples

### *Romeo and Juliet* Full Text Extraction

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes only and is not a finished product nor for medical advice.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Explore [Community Provider Plugins](COMMUNITY_PROVIDERS.md).  Learn to create your own with the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines.  You must sign a [Contributor License Agreement](https://cla.developers.google.com/about).

## Testing

```bash
pip install -e ".[test]"
pytest tests
```

or

```bash
tox
```

### Ollama Integration Testing

```bash
tox -e ollama-integration
```

## Development

### Code Formatting

```bash
./autoformat.sh
isort langextract tests --profile google --line-length 80
pyink langextract tests --config pyproject.toml
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Linting

```bash
pylint --rcfile=.pylintrc langextract tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Disclaimer

This is not an official Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).  For health-related applications, usage is subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**[Explore the LangExtract Repository](https://github.com/google/langextract)**