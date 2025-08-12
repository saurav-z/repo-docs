<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text Using LLMs

**Unleash the power of Large Language Models (LLMs) to transform unstructured text into structured, actionable data.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

**Key Features:**

*   🎯 **Precise Source Grounding:** Links extractions directly to the original text.
*   ✅ **Reliable Structured Outputs:** Guarantees consistent output schemas with few-shot examples.
*   🚀 **Optimized for Long Documents:** Efficiently handles large texts using chunking and parallel processing.
*   💡 **Interactive Visualization:** Generates self-contained HTML visualizations for easy review.
*   🌐 **Flexible LLM Support:** Works with cloud-based (Gemini, OpenAI) and local (Ollama) LLMs.
*   🧩 **Adaptable to Any Domain:** Customize extractions with simple examples.
*   🧠 **Leverages LLM World Knowledge:** Uses prompt engineering and examples to influence extraction accuracy.

## Table of Contents

*   [Introduction](#introduction)
*   [Why LangExtract?](#why-langextract)
*   [Quick Start](#quick-start)
*   [Installation](#installation)
    *   [From PyPI](#from-pypi)
    *   [From Source](#from-source)
    *   [Docker](#docker)
*   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
*   [Using OpenAI Models](#using-openai-models)
*   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
*   [More Examples](#more-examples)
    *   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    *   [Medication Extraction](#medication-extraction)
    *   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
*   [Contributing](#contributing)
    *   [Adding Custom Model Providers](#adding-custom-model-providers)
*   [Testing](#testing)
    *   [Ollama Integration Testing](#ollama-integration-testing)
*   [Development](#development)
    *   [Code Formatting](#code-formatting)
    *   [Pre-commit Hooks](#pre-commit-hooks)
    *   [Linting](#linting)
*   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents. It uses the power of LLMs to identify and organize key details within documents like clinical notes or reports, ensuring extracted data is accurately linked to its source.

## Why LangExtract?

LangExtract streamlines the process of information extraction with features like:

1.  **Precise Source Grounding:** Maps extractions to their exact locations for easy verification.
2.  **Reliable Structured Outputs:** Enforces output schema with few-shot learning for robust, structured results.
3.  **Optimized for Long Documents:** Handles large documents efficiently through chunking, parallel processing, and multi-pass extraction.
4.  **Interactive Visualization:** Generates interactive HTML files for in-context review.
5.  **Flexible LLM Support:** Supports various LLMs, from cloud to local open-source options.
6.  **Adaptable to Any Domain:** Define tasks using a few examples without model fine-tuning.
7.  **Leverages LLM World Knowledge:** Improve extraction accuracy using prompts and examples.

## Quick Start

> **Note:** Cloud model usage requires an API key; see [API Key Setup](#api-key-setup-for-cloud-models).

Extract structured information with just a few lines of code.

### 1. Define Your Extraction Task

Create a prompt describing what you want to extract, and provide a high-quality example:

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

Provide your input text and the prompt materials to the `lx.extract` function:

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

> **Model Selection**: `gemini-2.5-flash` is recommended. For complex tasks, consider `gemini-2.5-pro`. For production, use a Tier 2 Gemini quota. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2).
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
    f.write(html_content)
```

This creates an interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extraction close to the provided text. Prompts can be modified to use more LLM knowledge.

### Scaling to Longer Documents

Process full documents directly from URLs:

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

See the full *Romeo and Juliet* example for detailed results and performance: **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

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

LangExtract uses `pyproject.toml` for dependency management:

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

Cloud models (Gemini, OpenAI) require an API key. On-device models do not. For local LLMs, LangExtract supports Ollama.

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

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only for testing/development
)
```

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

LangExtract supports local inference with Ollama:

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

Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.
See [`examples/ollama/`](examples/ollama/) for detailed instructions.

## More Examples

### *Romeo and Juliet* Full Text Extraction

Extract information from *Romeo and Juliet* from Project Gutenberg (147,843 characters), showing parallel processing, sequential extraction passes, and performance optimization.
**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demo is for illustration only, and should not be used for medical advice.

Extract medical information from clinical text, demonstrating entity and relationship extraction.
**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live demo on HuggingFace Spaces:
**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to get started
You must sign a [Contributor License Agreement](https://cla.developers.google.com/about)
before submitting patches.

### Adding Custom Model Providers

LangExtract supports custom LLM providers through a plugin system. You can add support for new models by creating an external Python package that registers with LangExtract's provider registry. This allows you to:
- Add new model support without modifying the core library
- Distribute your provider independently
- Maintain custom dependencies

For detailed instructions, see the [Provider System Documentation](langextract/providers/README.md).

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

This project uses automated formatting tools:

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

Run linting:

```bash
pylint --rcfile=.pylintrc langextract tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Disclaimer

This is not an officially supported Google product. See the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, use is subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**
```

Key improvements and explanations:

*   **Concise Hook:**  The one-sentence hook immediately grabs attention and highlights the core benefit.
*   **Keyword Optimization:**  Uses relevant keywords like "structured data," "unstructured text," "LLMs," "information extraction," and model names throughout the README, including headings.
*   **Clear Headings and Structure:**  Organizes content logically with well-defined sections, making it easy to navigate.
*   **Bulleted Key Features:**  Uses bullets to quickly convey the main advantages of the library, improving readability and SEO.
*   **Actionable Quick Start:** The Quick Start is kept.
*   **Clearer Instructions:**  Improvements to the code examples with relevant commentary.
*   **Emphasis on Examples:** Prominently features the example links.
*   **Concise Explanations:**  Streamlined the explanations to be more direct and less verbose.
*   **Complete, Ready-to-Use:** The improved README is complete and self-contained.
*   **Conciseness:**  Removed unnecessary information.
*   **Direct Link Back to Repo:**  Added the link in the opening line.
*   **Emphasis on Model Options:** Clearly outlines various model options for user convenience.
*   **Removed redundant information:** The original README contained redundant information. This was removed.
*   **Enhanced Markdown:** Improved the markdown formatting.