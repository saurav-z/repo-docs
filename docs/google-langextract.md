<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Information from Unstructured Text

**Unlock the power of language models to transform raw text into structured data with LangExtract, a versatile Python library. [Explore the LangExtract GitHub Repository](https://github.com/google/langextract)!**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17015089.svg)](https://doi.org/10.5281/zenodo.17015089)

## Key Features

*   **Precise Source Grounding:**  Every extracted data point is directly linked to its origin in the source text, facilitating easy verification.
*   **Reliable Structured Outputs:** Leverage few-shot learning to enforce a consistent output schema, ensuring predictable, structured results.
*   **Optimized for Long Documents:** Tackle large documents efficiently through chunking, parallel processing, and iterative passes for improved recall.
*   **Interactive Visualization:**  Generate interactive HTML visualizations to review and explore extracted entities in their original context.
*   **Flexible LLM Support:**  Works with a wide range of LLMs, including cloud-based models (Gemini, OpenAI) and local open-source models via Ollama.
*   **Domain Agnostic:** Adaptable to any domain; simply define your extraction tasks with a few examples, without needing model fine-tuning.
*   **Leverages LLM Knowledge:**  Fine-tune extractions using precise prompts and few-shot examples to tap into the knowledge base of the chosen LLM.

## Table of Contents

-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup for Cloud Models](#api-key-setup-for-cloud-models)
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
-   [Disclaimer](#disclaimer)

## Quick Start

Quickly extract structured data using just a few lines of Python code.

> **Note:** Using cloud-hosted models such as Gemini requires an API key. Consult the [API Key Setup](#api-key-setup-for-cloud-models) section for details on obtaining and configuring your key.

### 1. Define Your Extraction Task

Create a prompt that clearly describes the desired extraction.  Provide a high-quality example to guide the model.

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

Use the `lx.extract` function to apply your prompt and example to the input text.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default. For more complex tasks, consider `gemini-2.5-pro`. For high-volume use, consider the Gemini Tier 2 quota (see [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2)).
>
> **Model Lifecycle**:  Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest stable and legacy model versions.

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

This creates an animated and interactive HTML file like:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:**  This example focuses on extracting from the text. The prompt can be modified to draw more on the LLM's knowledge base. The balance between evidence and inference depends on prompt instructions and example attributes.

### Scaling to Longer Documents

Process entire documents from URLs with parallel processing and enhanced sensitivity.

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

This extracts hundreds of entities from novels with high accuracy. The interactive visualization handles large result sets. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results and insights.

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

Use `pyproject.toml` for dependency management:

*Installing with `-e` enables development mode.*

```bash
git clone https://github.com/google/langextract.git
cd langextract

# Basic installation:
pip install -e .

# Development (includes linting):
pip install -e ".[dev]"

# Testing (includes pytest):
pip install -e ".[test]"
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

Cloud-hosted models (Gemini, OpenAI) require an API key.  On-device models do not.  LangExtract supports Ollama and can be extended to other third-party APIs.

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

Avoid directly including the API key in your code.

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

Extend LangExtract using a plugin system.

-   Add new model support independently
-   Distribute your provider as a separate package
-   Keep custom dependencies isolated
-   Override or extend built-in providers

See [Provider System Documentation](langextract/providers/README.md) for how to:

-   Register a provider with `@registry.register(...)`
-   Publish an entry point
-   Optionally provide a schema with `get_schema_class()`
-   Integrate with the factory via `create_model(...)`

## Using OpenAI Models

LangExtract supports OpenAI models (requires `pip install langextract[openai]`):

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

Run models locally using Ollama:

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

### *Romeo and Juliet* Full Text Extraction

Process full documents directly from URLs. This example extracts from the entire *Romeo and Juliet* text (147,843 characters), demonstrating parallel processing, sequential extraction passes, and performance optimization.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This is for illustrative purposes only and is not a finished or approved product. It is not intended for medical advice.

Extract structured medical information from clinical text (medication names, dosages, routes, and relationships).

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Explore RadExtract, a live interactive demo on HuggingFace Spaces:

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Discover community-created providers or add your own. See the [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry.

For plugin creation, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for development, testing, and pull request guidelines. You must sign a [Contributor License Agreement](https://cla.developers.google.com/about).

## Testing

Run tests locally:

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

Run integration tests with a local Ollama instance:

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

See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and, for health applications, the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).