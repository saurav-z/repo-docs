<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Unlock Structured Insights from Unstructured Text with LLMs

**Effortlessly transform unstructured text into structured data using the power of Language Models!**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17015089.svg)](https://doi.org/10.5281/zenodo.17015089)

## Key Features

*   **Precise Source Grounding:** Easily trace extractions back to their origin in the source text.
*   **Reliable Structured Outputs:** Enforce consistent output schemas using few-shot examples for structured results.
*   **Optimized for Long Documents:** Efficiently processes large documents with chunking, parallelization, and multiple passes for improved recall.
*   **Interactive Visualization:** Generate dynamic HTML files for easily reviewing extracted entities in context.
*   **Flexible LLM Support:**  Works with various LLMs including Google Gemini, OpenAI, and local models via Ollama.
*   **Domain Agnostic:** Adaptable to any domain – define extractions with just a few examples, no model fine-tuning needed.
*   **Leverages LLM Knowledge:** Utilize LLM world knowledge with precise prompts and few-shot examples.

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

Get started extracting structured information from text with just a few lines of code:

### 1. Define Your Extraction Task

Create a prompt and provide examples.

```python
import langextract as lx
import textwrap

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
```

### 2. Run the Extraction

Provide your input text, prompt, and examples.

```python
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

### 3. Visualize the Results

Save the results to a JSONL file and generate an interactive HTML visualization.

```python
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

**[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)** for detailed results and performance insights.

## Installation

### From PyPI

```bash
pip install langextract
```

*Consider a virtual environment for isolated installations.*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # On Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

```bash
git clone https://github.com/google/langextract.git
cd langextract

pip install -e .           # Basic install
pip install -e ".[dev]"    # Development install (includes linting)
pip install -e ".[test]"   # Testing install (includes pytest)
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

You'll need to set up an API key for cloud models like Gemini or OpenAI. Local LLMs via Ollama do not require an API key.

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
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
EOF
echo '.env' >> .gitignore  # Keep your API key secure
```

```python
import langextract as lx
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash"
)
```

**Option 3: Direct API Key (Not for Production)**

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

LangExtract has a lightweight plugin system. Add new model support without core code changes.

See the [Provider System Documentation](langextract/providers/README.md) for details.

## Using OpenAI Models

(Requires `pip install langextract[openai]`)

```python
import langextract as lx
import os

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

Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

## More Examples

### *Romeo and Juliet* Full Text Extraction

Process entire documents directly from URLs.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

Extract structured medical information from clinical text.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

Interactive demo on HuggingFace Spaces.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with custom model providers! Check out [Community Provider Plugins](COMMUNITY_PROVIDERS.md) or create your own.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for details.

## Testing

```bash
pip install -e ".[test]"
pytest tests  # Run all tests
```

Reproduce the CI matrix:
```bash
tox
```

Test Ollama integration (requires Ollama):
```bash
tox -e ollama-integration
```

## Development

### Code Formatting

```bash
./autoformat.sh  # Auto-format all code
isort langextract tests --profile google --line-length 80
pyink langextract tests --config pyproject.toml
```

### Pre-commit Hooks

```bash
pre-commit install
pre-commit run --all-files
```

### Linting

```bash
pylint --rcfile=.pylintrc langextract tests
```

## Disclaimer

This is not an officially supported Google product. See [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**[Visit the LangExtract Repository](https://github.com/google/langextract) for more details, examples, and contributions.**