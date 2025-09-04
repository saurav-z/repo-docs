<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**Unlock the power of Large Language Models (LLMs) to transform raw text into structured data with LangExtract, a Python library for information extraction.** Explore the original repo [here](https://github.com/google/langextract).

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17015089.svg)](https://doi.org/10.5281/zenodo.17015089)

## Key Features

*   **Precise Source Grounding:** Easily trace extracted information back to its original context.
*   **Reliable Structured Outputs:** Generate consistent, structured data using few-shot examples.
*   **Optimized for Long Documents:** Efficiently process large documents with chunking, parallel processing, and multiple passes.
*   **Interactive Visualization:** Instantly visualize extracted entities in their original context with an interactive HTML file.
*   **Flexible LLM Support:** Use your preferred LLMs, including Google Gemini, OpenAI models, and local LLMs with Ollama.
*   **Domain Agnostic:** Define extraction tasks for any domain using only a few examples, without model fine-tuning.

## Table of Contents

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
-   [Community Providers](#community-providers)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
    -   [Code Formatting](#code-formatting)
    -   [Pre-commit Hooks](#pre-commit-hooks)
    -   [Linting](#linting)
-   [Disclaimer](#disclaimer)

## Quick Start

Start extracting structured information with just a few lines of code:

### 1. Define Your Extraction Task

Create a prompt describing the extraction task and provide a high-quality example.

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

Use the `lx.extract` function with your input text, prompt, and example.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default. For more complex tasks, consider `gemini-2.5-pro`. For production, use a Tier 2 Gemini quota.  See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Gemini models have defined retirement dates; consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest versions.

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

This will create an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** The extraction example prioritizes text evidence. You can modify the task to leverage LLM knowledge, impacting the balance between text evidence and knowledge inference.

### Scaling to Longer Documents

Process large texts directly from URLs with parallel processing.

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

This approach extracts hundreds of entities accurately from full novels. The interactive visualization handles large result sets seamlessly. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results and performance.

## Installation

### From PyPI

```bash
pip install langextract
```

*For isolated environments:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

LangExtract uses `pyproject.toml` for dependency management:

*Installing with `-e` allows code modification without reinstalling.*

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

Required for cloud-hosted models (like Gemini or OpenAI), on-device models do not require API keys.

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

**Option 4: Vertex AI (Service Accounts)**

Use [Vertex AI](https://cloud.google.com/vertex-ai/docs/start/introduction-unified-platform) for service account authentication:

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

Extend LangExtract with custom LLM providers via a plugin system. See the [Provider System Documentation](langextract/providers/README.md) for details.

## Using OpenAI Models

(Requires `pip install langextract[openai]`)

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

(No API key required)

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

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`. See [`examples/ollama/`](examples/ollama/) for setup details.

## More Examples

### *Romeo and Juliet* Full Text Extraction

Demonstrates extracting from the full text of *Romeo and Juliet* (147,843 characters), using parallel processing and optimization.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

(Disclaimer: For illustrative purposes only; not a finished product or medical advice.)

Examples demonstrate entity recognition (medication names, dosages) and relationship extraction.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

A live interactive demo on HuggingFace Spaces.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Discover or contribute to the [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines. You must sign a [Contributor License Agreement](https://cla.developers.google.com/about).

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

Or use tox for the full CI matrix:

```bash
tox
```

### Ollama Integration Testing

Run if Ollama is installed:

```bash
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, usage is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).