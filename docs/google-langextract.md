<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Unlock Structured Insights from Unstructured Text with LLMs

**LangExtract is a powerful Python library that leverages Large Language Models (LLMs) to extract structured information from unstructured text, transforming raw data into actionable insights.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

**Key Features:**

*   ✅ **Precise Source Grounding:** Every extraction is linked to its exact location in the original text.
*   ✅ **Structured Outputs:** Enforces consistent, user-defined output schemas for reliable results.
*   ✅ **Optimized for Long Documents:** Efficiently handles large documents with chunking, parallel processing, and multi-pass extraction.
*   ✅ **Interactive Visualization:**  Generates self-contained HTML files for easy review and exploration of extracted entities.
*   ✅ **Flexible LLM Support:** Works with cloud-based LLMs (Gemini, OpenAI) and local open-source models (Ollama).
*   ✅ **Domain Agnostic:** Adaptable to any domain with just a few examples, no fine-tuning required.
*   ✅ **Leverages LLM Knowledge:** Utilize LLMs' world knowledge for enhanced information extraction, controlled by your prompt instructions and examples.

## Table of Contents

*   [Introduction](#introduction)
*   [Why LangExtract?](#why-langextract)
*   [Quick Start](#quick-start)
*   [Installation](#installation)
*   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
*   [Adding Custom Model Providers](#adding-custom-model-providers)
*   [Using OpenAI Models](#using-openai-models)
*   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
*   [More Examples](#more-examples)
    *   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    *   [Medication Extraction](#medication-extraction)
    *   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
*   [Contributing](#contributing)
*   [Testing](#testing)
*   [Development](#development)
*   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents using Large Language Models (LLMs). It excels at processing various materials like clinical notes and reports, identifying and organizing key details while ensuring traceability to the original text.

## Why LangExtract?

LangExtract offers a robust and flexible solution for information extraction with the following benefits:

1.  **Precise Source Grounding:**  Allows easy traceability and verification by mapping each extraction to its specific location in the source text, which can be visualized.
2.  **Reliable Structured Outputs:** Leverages few-shot examples to enforce a consistent output schema, producing robust, structured results with supported models like Gemini.
3.  **Optimized for Long Documents:** Uses text chunking, parallel processing, and multiple passes to efficiently extract data from large documents.
4.  **Interactive Visualization:** Generates a self-contained, interactive HTML file to visualize and review entities in their original context.
5.  **Flexible LLM Support:** Supports a wide range of LLMs, from cloud-based models (e.g., Google Gemini) to local open-source models via Ollama.
6.  **Adaptable to Any Domain:** Define extraction tasks for any domain using just a few examples, making it adaptable to various needs without model fine-tuning.
7.  **Leverages LLM World Knowledge:** Uses precise prompt wording and few-shot examples to harness the LLM's world knowledge, though the accuracy of extracted information depends on the selected LLM, task complexity, prompt clarity, and prompt examples.

## Quick Start

> **Note:**  Using cloud-hosted models like Gemini requires an API key.  See the [API Key Setup](#api-key-setup-for-cloud-models) section for instructions.

Quickly extract structured information with just a few lines of code:

### 1. Define Your Extraction Task

Create a prompt and guide the model with a clear example.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default, offering a balance of speed, cost, and quality.  For complex tasks, `gemini-2.5-pro` may be better.  Consider a Tier 2 Gemini quota for production use to avoid rate limits (see [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2)).
>
> **Model Lifecycle**: Gemini models have a lifecycle with retirement dates.  Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for updates.

### 3. Visualize the Results

Save the extractions to a `.jsonl` file and generate an interactive HTML visualization.

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

This creates an interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to text evidence. The task could be modified to generate attributes that draw more heavily from the LLM's world knowledge (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`). The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

Process entire documents from URLs with parallel processing and enhanced sensitivity:

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

This approach extracts hundreds of entities from full novels accurately. The interactive visualization handles large results.  **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for results and performance insights.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users. Consider a virtual environment:*

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

Cloud models (Gemini, OpenAI) require API keys. On-device models don't. LangExtract supports Ollama and can be extended to other third-party APIs.

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
    api_key="your-api-key-here"  # Only use this for testing/development
)
```

## Adding Custom Model Providers

LangExtract supports custom LLM providers via a lightweight plugin system. Add support for new models without core code changes.

- Add new model support independently
- Distribute your provider as a separate Python package
- Keep custom dependencies isolated
- Override or extend built-in providers via priority

See [Provider System Documentation](langextract/providers/README.md) for details:

- Register a provider with `@registry.register(...)`
- Publish an entry point for discovery
- Optionally provide a schema with `get_schema_class()`
- Integrate with the factory via `create_model(...)`

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

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

See [`examples/ollama/`](examples/ollama/) for detailed setup and examples.

## More Examples

Additional examples of LangExtract in action:

### *Romeo and Juliet* Full Text Extraction

Process the complete *Romeo and Juliet* text from Project Gutenberg (147,843 characters), showcasing parallel processing, sequential extraction passes, and performance optimization.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:**  This is for illustrative purposes and not a finished product.  It is not for medical advice.

Demonstrates extracting structured medical information from clinical text.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live demo on HuggingFace Spaces. Automatically structure radiology reports.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to start. Sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

## Testing

To run tests:

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

This is not an official Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).  For health-related applications, usage is subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**  [Explore the LangExtract Repository](https://github.com/google/langextract)