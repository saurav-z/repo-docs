<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text Using LLMs

**Unlock the power of LLMs to transform raw text into structured data with LangExtract, a Python library designed for robust, accurate, and efficient information extraction.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Source Grounding:**  Every extracted piece of information is linked directly to its source text, simplifying verification and highlighting.
*   **Reliable Structured Outputs:** Enforces a consistent output schema based on your few-shot examples, guaranteeing robust, structured results using supported models like Gemini.
*   **Optimized for Long Documents:** Handles lengthy documents efficiently using text chunking, parallel processing, and multi-pass extraction.
*   **Interactive Visualization:** Generates self-contained HTML files to visualize extractions and their original context.
*   **Flexible LLM Support:** Works with cloud-based LLMs (Google Gemini family, OpenAI) and local open-source models via Ollama.
*   **Domain Agnostic:**  Define extraction tasks for any domain with a few examples.
*   **Leverages LLM Knowledge:** Infers additional data based on your prompt instructions and example attributes, and utilizes LLM's world knowledge to improve output quality.

## Table of Contents

-   [Introduction](#introduction)
-   [Why LangExtract?](#why-langextract)
-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
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

LangExtract is a Python library designed to extract structured information from unstructured text documents using Large Language Models (LLMs). It's built to process various text formats, like clinical notes or reports, and excels at identifying key details and organizing them according to user-defined instructions. LangExtract ensures that extracted data accurately corresponds to the original text.

## Why LangExtract?

LangExtract offers several advantages for information extraction:

1.  **Precise Source Grounding:** Tracks the exact location of each extraction in the source text.
2.  **Reliable Structured Outputs:** Generates consistent output based on provided examples.
3.  **Optimized for Long Documents:** Efficiently processes large documents using chunking, parallel processing, and multiple passes.
4.  **Interactive Visualization:** Creates self-contained, interactive HTML files to review extracted entities.
5.  **Flexible LLM Support:** Supports cloud and local LLMs.
6.  **Adaptable to Any Domain:** Define extraction tasks with a few examples.
7.  **Leverages LLM World Knowledge:** Improves accuracy by using LLM knowledge.

## Quick Start

> **Note:** Cloud-hosted models (Gemini, OpenAI) need an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section.

Extract information in just a few code lines.

### 1. Define Your Extraction Task

Create a prompt that clearly explains what you want to extract, and provide a high-quality example:

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

Use the `lx.extract` function with your input text and prompt:

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

> **Model Selection**:  `gemini-2.5-flash` provides a good balance of speed, cost, and quality. `gemini-2.5-pro` is recommended for complex tasks.  Consider a Tier 2 Gemini quota for production.  Refer to the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for version details.

### 3. Visualize the Results

Save extractions to a `.jsonl` file and generate an interactive HTML visualization:

```python
# Save the results to a JSONL file
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

# Generate the visualization from the file
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    f.write(html_content)
```

This creates an interactive HTML file (example below):

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:**  The extraction accuracy depends on your prompt instructions. You can adjust the example attributes to make use of the LLM's world knowledge (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`).

### Scaling to Longer Documents

Process documents directly from URLs with parallel processing and enhanced sensitivity:

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

The interactive visualization handles large result sets effectively.  **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for details.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users. Consider using a virtual environment for isolated environments:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

Uses `pyproject.toml` for dependency management:

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

You'll need an API key to use LangExtract with cloud models (Gemini, OpenAI). On-device models (Ollama) don't require an API key.

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

Not recommended for production use:

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

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False` because LangExtract doesn't implement schema constraints for OpenAI yet.

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

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

For detailed installation, Docker setup, and examples, see [`examples/ollama/`](examples/ollama/).

## More Examples

Find more LangExtract examples:

### *Romeo and Juliet* Full Text Extraction

Process complete documents directly from URLs.  Example uses the full text of *Romeo and Juliet* from Project Gutenberg, demonstrating parallel processing and performance optimization.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:**  This is for illustrative purposes only and is not a finished product. It should not be used for medical advice.

Extract structured medical information from clinical text, including medication names, dosages, routes, and relationships.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Try RadExtract, a live demo on HuggingFace Spaces, to automatically structure radiology reports directly in your browser.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for details on development, testing, and pull requests. You must sign a
[Contributor License Agreement](https://cla.developers.google.com/about).

### Adding Custom Model Providers

LangExtract supports custom LLM providers through a plugin system. Create an external Python package that registers with LangExtract's provider registry to:

*   Add new model support without modifying the core library
*   Distribute your provider independently
*   Maintain custom dependencies

For instructions, see the [Provider System Documentation](langextract/providers/README.md).

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

This is not an officially supported Google product. Please cite the library and acknowledge usage if using it in production or publications. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) for health-related applications.

---

**Get started today and easily extract valuable insights from your text data with [LangExtract](https://github.com/google/langextract)!**