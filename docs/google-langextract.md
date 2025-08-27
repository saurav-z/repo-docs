<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Extract Structured Data from Unstructured Text with LLMs

**Effortlessly transform raw text into structured insights using the power of Large Language Models (LLMs).**  [See the original repo](https://github.com/google/langextract).

**Key Features:**

*   ✅ **Precise Source Grounding:** Pinpoint extractions to their exact location in the source text for easy verification.
*   ✅ **Reliable Structured Outputs:** Consistent, schema-based results using few-shot examples.
*   ✅ **Optimized for Long Documents:** Efficiently processes large texts using chunking, parallelization, and multiple passes.
*   ✅ **Interactive Visualization:**  Generate interactive HTML visualizations to review and explore extractions in context.
*   ✅ **Flexible LLM Support:** Works with a range of models, from cloud-based (Gemini, OpenAI) to local open-source LLMs via Ollama.
*   ✅ **Domain Agnostic:** Adaptable to any domain with just a few examples – no model fine-tuning required.
*   ✅ **Leverages LLM Knowledge:**  Infuse world knowledge into extractions through prompt engineering.

## Table of Contents

-   [Introduction](#introduction)
-   [Key Benefits](#key-benefits)
-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup](#api-key-setup-for-cloud-models)
-   [Adding Custom Model Providers](#adding-custom-model-providers)
-   [Using OpenAI Models](#using-openai-models)
-   [Using Local LLMs with Ollama](#using-local-llms-with-ollama)
-   [Advanced Examples](#more-examples)
    -   [*Romeo and Juliet* Full Text Extraction](#romeo-and-juliet-full-text-extraction)
    -   [Medication Extraction](#medication-extraction)
    -   [Radiology Report Structuring: RadExtract](#radiology-report-structuring-radextract)
-   [Community Providers](#community-providers)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
-   [Disclaimer](#disclaimer)

## Introduction

LangExtract is a Python library designed to extract structured information from unstructured text documents using the power of LLMs. Ideal for processing clinical notes, reports, and other text-heavy materials, LangExtract identifies and organizes key details based on user-defined instructions.  It ensures that the extracted data is accurately linked back to its source, making the process transparent and reliable.

## Quick Start

Get started extracting data in just a few steps.

### 1. Define Your Extraction Task

Create a clear prompt outlining the information you want to extract, followed by a high-quality example to guide the model.

```python
import langextract as lx
import textwrap

# Define the prompt and extraction rules
prompt = textwrap.dedent("""\
    Extract characters, emotions, and relationships in order of appearance.
    Use exact text for extractions. Do not paraphrase or overlap entities.
    Provide meaningful attributes for each entity to add context.""")

# Provide a high-quality example
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

Use the `lx.extract` function with your input text, prompt, and examples.

```python
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**:  `gemini-2.5-flash` is the recommended default for speed and cost.  For complex tasks, consider `gemini-2.5-pro`.  For production, a Tier 2 Gemini quota is suggested. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.
>
> **Model Lifecycle**: Gemini models have a lifecycle.  Refer to the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) for the latest versions.

### 3. Visualize the Results

Save the extractions to a `.jsonl` file and then generate an interactive HTML visualization.

```python
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")

html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

This generates an interactive visualization:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

> **Note on LLM Knowledge Utilization:** The prompt can be tailored to extract text evidence or draw more heavily on LLM knowledge. The balance is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

Process entire documents using URLs with parallel processing.

```python
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

The interactive visualization handles large result sets.  **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

## Installation

### From PyPI

```bash
pip install langextract
```

*For isolated environments:*

```bash
python -m venv langextract_env
source langextract_env/bin/activate  # Windows: langextract_env\Scripts\activate
pip install langextract
```

### From Source

```bash
git clone https://github.com/google/langextract.git
cd langextract

# Basic installation:
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

Set up an API key for cloud-hosted models (Gemini, OpenAI). On-device models don't require a key.

### API Key Sources

*   [AI Studio](https://aistudio.google.com/app/apikey) (Gemini)
*   [Vertex AI](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) (Enterprise)
*   [OpenAI Platform](https://platform.openai.com/api-keys) (OpenAI)

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

**Option 3: Direct API Key (Not Recommended)**

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # For testing/development only
)
```

## Adding Custom Model Providers

Extend LangExtract with a lightweight plugin system.

- Add new model support without core code changes.
- Distribute your provider as a separate Python package.
- Keep custom dependencies isolated.
- Override or extend built-in providers via priority-based resolution.

See the [Provider System Documentation](langextract/providers/README.md) for details.

## Using OpenAI Models

LangExtract supports OpenAI models (requires `pip install langextract[openai]`):

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",  # Selects OpenAI provider
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

Note: OpenAI models require `fence_output=True` and `use_schema_constraints=False`.

## Using Local LLMs with Ollama

LangExtract supports local inference using Ollama:

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",  # Selects Ollama provider
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False
)
```

**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

For details, see [`examples/ollama/`](examples/ollama/).

## Advanced Examples

### *Romeo and Juliet* Full Text Extraction

Demonstrates extracting from the full text of *Romeo and Juliet* from Project Gutenberg.  Shows parallel processing, sequential extraction passes, and performance optimization for long document processing.

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

> **Disclaimer:**  For illustrative purposes only. Not a finished product, not for medical advice.

Extract structured medical information. Examples of basic entity and relationship extraction.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

A live interactive demo on HuggingFace Spaces that shows how LangExtract can automatically structure radiology reports.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Extend LangExtract with custom model providers! Check out our [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry to discover providers created by the community or add your own.

For detailed instructions on creating a provider plugin, see the [Custom Provider Plugin Example](examples/custom_provider_plugin/).

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for development guidelines. You must sign a
[Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

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
tox  # runs pylint + pytest
```

### Ollama Integration Testing

```bash
# Test Ollama integration (requires Ollama running)
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).  For health-related applications, usage is subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).