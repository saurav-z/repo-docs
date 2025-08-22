<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Text using LLMs

**Unlock the power of language models to transform unstructured text into structured, actionable insights with LangExtract.**

[Visit the LangExtract GitHub Repository](https://github.com/google/langextract)

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Source Grounding:**  Links extracted data directly to its source text, enabling easy verification.
*   **Reliable Structured Outputs:** Guarantees consistent output schemas using few-shot examples and supported LLMs.
*   **Optimized for Long Documents:** Handles large documents efficiently with chunking, parallel processing, and multi-pass extraction.
*   **Interactive Visualization:** Generates interactive HTML visualizations for easy review of extracted entities in context.
*   **Flexible LLM Support:** Supports cloud-based (e.g., Gemini) and local open-source LLMs (via Ollama).
*   **Domain Agnostic:** Adaptable to any domain – define extraction tasks using just a few examples.
*   **Leverages LLM Knowledge:**  Utilize LLM world knowledge through prompt engineering and examples.

## Table of Contents

-   [Introduction](#introduction)
-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup](#api-key-setup-for-cloud-models)
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
-   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents using the power of Large Language Models (LLMs).  It allows users to define extraction rules and identify key details within documents like clinical notes or reports, structuring and organizing the extracted data while maintaining links to the source text.

## Quick Start

Extract structured information with just a few lines of code.

> **Note:** Using cloud-hosted models like Gemini requires an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section for instructions.

### 1. Define Your Extraction Task

First, create a prompt that clearly describes what you want to extract. Then, provide a high-quality example to guide the model.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default. For more complex tasks, `gemini-2.5-pro` can provide superior results. For large-scale use, consider a Tier 2 Gemini quota for increased throughput.  See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2).
>
> **Model Lifecycle**: Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest stable and legacy Gemini versions.

### 3. Visualize the Results

Save extractions to a `.jsonl` file and then generate an interactive HTML visualization.

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

> **Note on LLM Knowledge Utilization:**  The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

Process entire documents directly from URLs using parallel processing:

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

The interactive visualization makes it easy to explore results. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users.*

For virtual environments:

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses `pyproject.toml` for dependency management:

*Installing with `-e` allows code modifications without reinstallation.*

```bash
git clone https://github.com/google/langextract.git
cd langextract

# Basic installation:
pip install -e .

# Development installation (includes linting tools):
pip install -e ".[dev]"

# Testing installation (includes pytest):
pip install -e ".[test]"
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

When using cloud-hosted models (Gemini, OpenAI), you'll need an API key.  On-device models (like Ollama) do not require an API key.

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
    api_key="your-api-key-here"  # Only use for testing/development
)
```

## Adding Custom Model Providers

LangExtract supports custom LLM providers via a lightweight plugin system. Add support for new models independently.

- Add new model support independently of the core library
- Distribute your provider as a separate Python package
- Keep custom dependencies isolated
- Override or extend built-in providers via priority-based resolution

See [Provider System Documentation](langextract/providers/README.md) for how to:

-   Register a provider with `@registry.register(...)`
-   Publish an entry point
-   Optionally provide a schema with `get_schema_class()`
-   Integrate with the factory via `create_model(...)`

## Using OpenAI Models

LangExtract supports OpenAI models (requires optional dependency: `pip install langextract[openai]`):

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

LangExtract supports local inference using Ollama (no API keys required):

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

For detailed setup, see [`examples/ollama/`](examples/ollama/).

## More Examples

### *Romeo and Juliet* Full Text Extraction

Process the complete text of *Romeo and Juliet* from Project Gutenberg (147,843 characters) using parallel processing and multi-pass extraction.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This is for illustrative purposes only and not for medical advice.

Extract structured medical information from clinical text. Examples demonstrate entity and relationship extraction.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore a live demo on HuggingFace Spaces that structures radiology reports.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for development guidelines. You must sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

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

If you have Ollama installed:

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

This is not an officially supported Google product.  Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and, for health-related applications, the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).