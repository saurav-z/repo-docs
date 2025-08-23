<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**LangExtract is a powerful Python library that uses Large Language Models (LLMs) to intelligently extract and organize structured information from text documents, making data extraction simpler than ever.**

## Key Features

*   **Precise Source Grounding:** Links extracted information directly to the original text, ensuring traceability.
*   **Reliable Structured Outputs:**  Enforces consistent, structured output schemas based on your specifications.
*   **Optimized for Long Documents:**  Effectively processes large documents using chunking, parallelization, and multi-pass extraction.
*   **Interactive Visualization:**  Generates interactive HTML visualizations to easily review and validate extracted entities within their context.
*   **Flexible LLM Support:**  Supports a variety of LLMs, from cloud-based services like Gemini to local open-source models via Ollama.
*   **Domain Agnostic:**  Adaptable to any domain with customizable extraction tasks defined by examples.
*   **Leverages LLM Knowledge:** Harnesses LLM knowledge to enrich extractions (influenced by your prompts and examples).

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

## Quick Start

Get started extracting structured data in just a few steps:

```python
import langextract as lx
import textwrap

# Define the extraction task with a prompt and example
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

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

# Process input text and run the extraction
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

# Visualize Results
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

*   **Model Selection**: `gemini-2.5-flash` is the recommended default for speed, cost, and quality. Consider `gemini-2.5-pro` for more complex tasks, and consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest versions.
*   **Scaling to Longer Documents:**  Process large texts from URLs using parallel processing and extraction passes.  See the full *Romeo and Juliet* example for more details.

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

```bash
git clone https://github.com/google/langextract.git
cd langextract

# Basic Installation:
pip install -e .

# Development Installation (includes linting):
pip install -e ".[dev]"

# Testing Installation:
pip install -e ".[test]"
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

LangExtract requires API keys for cloud-hosted LLMs like Gemini or OpenAI. Local models do not require an API key.

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

**Option 3: Direct API Key (Not Recommended)**

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only for testing/development
)
```

## Adding Custom Model Providers

Extend LangExtract with custom LLM providers via a lightweight plugin system. See the [Provider System Documentation](langextract/providers/README.md) for details:

*   Register a provider with `@registry.register(...)`
*   Publish an entry point for discovery
*   Optionally provide a schema with `get_schema_class()`
*   Integrate with the factory via `create_model(...)`

## Using OpenAI Models

Install the OpenAI dependency (`pip install langextract[openai]`) and use OpenAI models:

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

Run local inference with Ollama:

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

*   Install Ollama from [ollama.com](https://ollama.com/) and run `ollama pull gemma2:2b`.
*   See [`examples/ollama/`](examples/ollama/) for more details.

## More Examples

Explore LangExtract's capabilities through these examples:

### *Romeo and Juliet* Full Text Extraction

Process the complete text of *Romeo and Juliet* (147,843 characters) using parallel processing and extraction passes.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

Extract structured medical information from clinical text.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

An interactive demo using LangExtract to structure radiology reports on HuggingFace Spaces.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contribute to LangExtract by following the guidelines in [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md). You must sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

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

See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for detailed development guidelines.

## Disclaimer

This is not an officially supported Google product. See the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms) for more information.

---

**[Explore the LangExtract Repo](https://github.com/google/langextract)**