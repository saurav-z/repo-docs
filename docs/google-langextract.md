<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**Unlock the power of LLMs to transform unstructured text into structured data with LangExtract, a versatile Python library for information extraction.**  ([View Original Repo](https://github.com/google/langextract))

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   **Precise Extraction:** Map extractions directly to their source text for easy verification.
*   **Structured Output:** Enforce a consistent schema for reliable, structured results.
*   **Optimized for Long Documents:** Efficiently processes large documents using chunking, parallel processing, and multi-pass extraction.
*   **Interactive Visualization:** Generate self-contained HTML visualizations for easy review and analysis.
*   **Flexible LLM Support:**  Supports a range of models, including Google Gemini and local models via Ollama.
*   **Domain Agnostic:**  Define extraction tasks for any domain with just a few examples.
*   **Leverages LLM Knowledge:** Utilize prompt wording and few-shot examples to influence how the extraction task may utilize LLM knowledge.

## Table of Contents

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
-   [Community Providers](#community-providers)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
-   [Disclaimer](#disclaimer)

## Quick Start

Get started extracting structured data in just a few steps!  Remember to set up your API key if using cloud-based models (see [API Key Setup](#api-key-setup-for-cloud-models)).

### 1. Define Your Extraction Task

Craft a prompt that clearly defines what you want to extract.  Provide a few high-quality examples to guide the model.

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

Pass your input text, prompt, and examples to the `lx.extract` function.

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

> **Model Selection**: `gemini-2.5-flash` is the recommended default, offering an excellent balance of speed, cost, and quality. For highly complex tasks requiring deeper reasoning, `gemini-2.5-pro` may provide superior results. For large-scale or production use, a Tier 2 Gemini quota is suggested to increase throughput and avoid rate limits. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Note that Gemini models have a lifecycle with defined retirement dates. Users should consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed about the latest stable and legacy versions.

### 3. Visualize the Results

Save your extractions to a `.jsonl` file and generate an interactive HTML visualization to review your entities in context.

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

This produces an interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to the text evidence - extracting "longing" for Lady Juliet's emotional state and identifying "yearning" from "gazed longingly at the stars." The task could be modified to generate attributes that draw more heavily from the LLM's world knowledge (e.g., adding `"identity": "Capulet family daughter"` or `"literary_context": "tragic heroine"`). The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

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

Explore the full *Romeo and Juliet* extraction example for detailed results and performance insights:  **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

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

When using cloud-hosted models (like Gemini or OpenAI), you'll need to set up an API key.  On-device models do not require an API key.

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

Extend LangExtract's capabilities by adding custom LLM providers.  The lightweight plugin system lets you integrate new models without modifying core code.

-   Add new model support independently of the core library.
-   Distribute your provider as a separate Python package.
-   Keep custom dependencies isolated.
-   Override or extend built-in providers via priority-based resolution.

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

LangExtract supports local inference using Ollama, allowing you to run models without API keys:

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

Explore advanced use cases with these examples:

### *Romeo and Juliet* Full Text Extraction

Process complete documents directly from URLs. This example extracts from the full text of *Romeo and Juliet* (147,843 characters), demonstrating parallel processing and optimization for long documents.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:** This demonstration is for illustrative purposes of LangExtract's baseline capability only. It does not represent a finished or approved product, is not intended to diagnose or suggest treatment of any disease or condition, and should not be used for medical advice.

Extract structured medical information from clinical text, including entity and relationship extraction.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

See how LangExtract can automatically structure radiology reports. Try the live interactive demo on HuggingFace Spaces.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with community-created model providers! Discover and contribute to the growing list of plugins in the [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry.

For instructions on creating a provider plugin, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for development, testing, and pull request guidelines.  You must sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

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

This test will automatically detect if Ollama is available and run real inference tests.

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

Run linting before submitting PRs:

```bash
pylint --rcfile=.pylintrc langextract tests
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for detailed development guidelines.

## Disclaimer

This is not an officially supported Google product.  Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).  For health-related applications, use is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**