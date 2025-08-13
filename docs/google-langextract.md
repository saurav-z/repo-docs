<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text with LLMs

**Unlock the power of Large Language Models (LLMs) to transform unstructured text into structured, actionable data with LangExtract!**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Source Grounding:** Ensures every extraction is linked to its original text location for easy verification.
*   **Reliable Structured Outputs:** Leverages few-shot examples to enforce consistent output formats, reducing errors.
*   **Optimized for Long Documents:** Efficiently processes large texts with chunking, parallel processing, and multiple passes.
*   **Interactive Visualization:** Generates interactive HTML files for easy exploration and review of extracted entities.
*   **Flexible LLM Support:** Works with various LLMs, including Google Gemini, OpenAI, and local models through Ollama.
*   **Domain Agnostic:** Adaptable to any domain through user-defined prompts and examples – no fine-tuning required!
*   **Leverages LLM Knowledge:** Effectively uses LLM's world knowledge and reasoning for more insightful extractions.

## Table of Contents

*   [Introduction](#introduction)
*   [Why LangExtract?](#why-langextract)
*   [Quick Start](#quick-start)
*   [Installation](#installation)
    *   [From PyPI](#from-pypi)
    *   [From Source](#from-source)
    *   [Docker](#docker)
*   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
    *   [API Key Sources](#api-key-sources)
    *   [Setting up API key in your environment](#setting-up-api-key-in-your-environment)
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

LangExtract is a powerful Python library designed to extract structured information from unstructured text documents using LLMs. It excels at identifying and organizing key details, while ensuring extracted data is directly tied to the source text.

## Why LangExtract?

(See original README for this section)

## Quick Start

(See original README for this section)

### 1. Define Your Extraction Task

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default...
>
> **Model Lifecycle**: Note that Gemini models have a lifecycle with defined retirement dates...

### 3. Visualize the Results

```python
# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content)
```

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to the text evidence...

### Scaling to Longer Documents

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

This approach can extract hundreds of entities from full novels while maintaining high accuracy. The interactive visualization seamlessly handles large result sets, making it easy to explore hundreds of entities from the output JSONL file. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results and performance insights.

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

(See original README for this section)

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

(See original README for this section)

### API Key Sources

(See original README for this section)

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

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False` because LangExtract doesn't implement schema constraints for OpenAI yet.

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

For detailed installation, Docker setup, and examples, see [`examples/ollama/`](examples/ollama/).

## More Examples

### *Romeo and Juliet* Full Text Extraction

LangExtract can process complete documents directly from URLs. This example demonstrates extraction from the full text of *Romeo and Juliet* from Project Gutenberg (147,843 characters), showing parallel processing, sequential extraction passes, and performance optimization for long document processing.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes of LangExtract's baseline capability only...

LangExtract excels at extracting structured medical information from clinical text. These examples demonstrate both basic entity recognition (medication names, dosages, routes) and relationship extraction (connecting medications to their attributes), showing LangExtract's effectiveness for healthcare applications.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live interactive demo on HuggingFace Spaces that shows how LangExtract can automatically structure radiology reports. Try it directly in your browser with no setup required.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to get started
with development, testing, and pull requests. You must sign a
[Contributor License Agreement](https://cla.developers.google.com/about)
before submitting patches.

### Adding Custom Model Providers

LangExtract supports custom LLM providers through a plugin system...

For detailed instructions, see the [Provider System Documentation](langextract/providers/README.md).

## Testing

To run tests locally from the source:

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

If you have Ollama installed locally, you can run integration tests:

```bash
# Test Ollama integration (requires Ollama running with gemma2:2b model)
tox -e ollama-integration
```

This test will automatically detect if Ollama is available and run real inference tests.

## Development

### Code Formatting

(See original README for this section)

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

This is not an officially supported Google product. If you use
LangExtract in production or publications, please cite accordingly and
acknowledge usage. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).
For health-related applications, use of LangExtract is also subject to the
[Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**

**[Back to Top](#langextract-effortlessly-extract-structured-data-from-unstructured-text-with-llms)** (Links to the top of the README)