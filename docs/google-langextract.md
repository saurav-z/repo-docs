<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text with LLMs

**Unlock the power of Large Language Models (LLMs) to transform unstructured text into structured, actionable data.**  LangExtract empowers you to extract key information from any text source, making it easy to analyze, visualize, and integrate data into your workflows.

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Source Grounding:** Links extracted information directly to the original text for easy verification.
*   **Consistent Structured Outputs:**  Leverages few-shot examples to enforce a consistent output schema, ensuring reliable results.
*   **Optimized for Long Documents:**  Handles large documents with chunking, parallel processing, and multiple passes for improved recall.
*   **Interactive Visualization:**  Generates interactive HTML visualizations to easily review and understand extracted data.
*   **Flexible LLM Support:** Works with both cloud-based models (e.g., Gemini) and local LLMs via Ollama.
*   **Domain Agnostic:** Easily adaptable to any domain with just a few examples; no model fine-tuning required.
*   **Leverages LLM Knowledge:** Utilizes precise prompts and examples to tap into LLM world knowledge for richer extractions.

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
*   [Community Providers](#community-providers)
*   [Contributing](#contributing)
*   [Testing](#testing)
*   [Development](#development)
*   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents. It employs the power of LLMs to identify, organize, and extract key details based on user-defined instructions. Whether you're working with clinical notes, reports, or other textual data, LangExtract can help you unlock valuable insights.

## Why LangExtract?

*   **Accuracy and Traceability:**  Ensure extracted data is grounded in the source text for increased reliability.
*   **Reliable Structure:**  Benefit from consistent and well-defined output schemas, thanks to controlled generation.
*   **Efficiency and Scalability:**  Handle long documents efficiently with optimized processing techniques.
*   **Insightful Visualization:** Quickly analyze and review thousands of entities with interactive visualizations.
*   **Model Agnosticism:** Supports a wide range of LLMs, allowing you to choose the best fit for your needs.
*   **Rapid Domain Adaptation:**  Easily customize extractions for any domain without the need for model fine-tuning.
*   **Knowledge Integration:** Leverage the extensive world knowledge of LLMs to enhance extraction results.

## Quick Start

Get started with LangExtract in just a few steps!

>   **Note:** Using cloud-hosted models like Gemini requires an API key. See the [API Key Setup](#api-key-setup-for-cloud-models) section for instructions.

### 1. Define Your Extraction Task

Create a prompt that clearly describes the information you want to extract, along with a high-quality example to guide the model.

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

Pass your input text and prompt materials to the `lx.extract` function.

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

>   **Model Recommendations**: `gemini-2.5-flash` is the recommended default, offering a great balance of speed, cost, and quality.  For more complex tasks, consider `gemini-2.5-pro`.  For production use, a Tier 2 Gemini quota is suggested.  See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for more details.
>
>   **Model Lifecycle:** Gemini models have lifecycles with defined retirement dates. Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest updates.

### 3. Visualize the Results

Save the extraction results to a `.jsonl` file and generate an interactive HTML visualization.

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

This creates an interactive HTML file, as shown in this example:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

>   **LLM Knowledge Note:** This example extracts information directly from the text.  Modify the task to utilize the LLM's world knowledge, by changing the prompt's instructions or the attribute specifications.

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

This approach extracts hundreds of entities from full novels while maintaining high accuracy. The interactive visualization makes it easy to explore large datasets from the JSONL output file. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for details.

## Installation

### From PyPI

```bash
pip install langextract
```

*Recommended for most users. Consider using a virtual environment:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses modern Python packaging with `pyproject.toml` for dependency management:

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

Set up API keys when using cloud-hosted models like Gemini or OpenAI. On-device models don't require an API key.

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

You can also provide the API key directly in your code, though this is not recommended for production use:

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

Extend LangExtract by creating custom LLM providers using a lightweight plugin system. Add support for new models without changing the core code.

*   Add new model support separately from the core library
*   Distribute your provider as a separate Python package
*   Keep custom dependencies isolated
*   Override or extend built-in providers via priority-based resolution

See the detailed guide in [Provider System Documentation](langextract/providers/README.md) to learn how to:

-   Register a provider with `@registry.register(...)`
-   Publish an entry point for discovery
-   Optionally provide a schema with `get_schema_class()` for structured output
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

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False` because LangExtract doesn't implement schema constraints for OpenAI yet.

## Using Local LLMs with Ollama

LangExtract supports local inference using Ollama, letting you run models without API keys:

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

Explore more LangExtract examples:

### *Romeo and Juliet* Full Text Extraction

Process the entire text of *Romeo and Juliet* from Project Gutenberg (147,843 characters), showcasing parallel processing and performance optimization for long documents.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

>   **Disclaimer:** For illustrative purposes only.  Not a finished or approved product. Not for medical advice.

Demonstrates extracting structured medical information from clinical text, including medication names, dosages, and relationships.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live interactive demo on HuggingFace Spaces that shows how LangExtract can automatically structure radiology reports. Try it directly in your browser with no setup required.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with community-created model providers!  Check out the [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry.

For details on creating a provider plugin, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for development, testing, and pull requests.  You must sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

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

If you have Ollama installed locally, you can run integration tests:

```bash
# Test Ollama integration (requires Ollama running with gemma2:2b model)
tox -e ollama-integration
```

This test will automatically detect if Ollama is available.

## Development

### Code Formatting

Automated formatting tools maintain consistent code style:

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

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).
For health-related applications, use of LangExtract is also subject to the
[Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).