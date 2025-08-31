<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text with LLMs

**Unleash the power of Large Language Models to transform raw text into structured, actionable insights, making information extraction easy and efficient.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features

*   ✅ **Precise & Grounded Extractions:** Every extracted piece of information is linked to its exact source text, ensuring traceability.
*   ✅ **Robust Structured Outputs:** Leverage the power of few-shot examples to enforce a consistent output schema and generate reliable results, supported by models like Gemini.
*   ✅ **Optimized for Long Documents:** Tackle large documents efficiently with text chunking, parallel processing, and multiple passes for superior recall.
*   ✅ **Interactive Visualization:** Instantly visualize and review extracted entities within their original context using a self-contained HTML file.
*   ✅ **Flexible LLM Support:** Supports a range of models, including Google Gemini, OpenAI, and local open-source models via Ollama.
*   ✅ **Domain Agnostic:** Define extraction tasks for any domain using just a few examples, adapting without model fine-tuning.
*   ✅ **Leverages LLM World Knowledge:** Influence extractions by fine-tuning prompts to incorporate LLM world knowledge.

## Table of Contents

-   [Quick Start](#quick-start)
-   [Installation](#installation)
-   [API Key Setup](#api-key-setup-for-cloud-models)
-   [Adding Custom Model Providers](#adding-custom-model-providers)
-   [Examples](#more-examples)
-   [Community Providers](#community-providers)
-   [Contributing](#contributing)
-   [Testing](#testing)
-   [Development](#development)
-   [Disclaimer](#disclaimer)

## Quick Start

Effortlessly extract structured information with just a few lines of code.

### 1. Define Your Extraction Task

Create a prompt and high-quality example(s).

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

Provide your input text and prompt materials to the `lx.extract` function.

```python
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**: `gemini-2.5-flash` is the recommended default. For complex tasks, `gemini-2.5-pro` may be preferable.  For larger production use, a Tier 2 Gemini quota is suggested. See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2)
>
> **Model Lifecycle**: Consult the [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed about the latest stable and legacy versions.

### 3. Visualize the Results

The extractions can be saved to a `.jsonl` file and visualized in an interactive HTML file.

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

> **Note on LLM Knowledge Utilization:** This example demonstrates extractions that stay close to the text evidence. The task could be modified to generate attributes that draw more heavily from the LLM's world knowledge. The balance between text-evidence and knowledge-inference is controlled by your prompt instructions and example attributes.

### Scaling to Longer Documents

For larger texts, you can process entire documents directly from URLs with parallel processing and enhanced sensitivity:

```python
result = lx.extract(
    text_or_documents="https://www.gutenberg.org/files/1513/1513-0.txt",
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
    extraction_passes=3,
    max_workers=20,
    max_char_buffer=1000
)
```

The interactive visualization seamlessly handles large result sets. **[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

## Installation

### From PyPI

```bash
pip install langextract
```

### From Source

```bash
git clone https://github.com/google/langextract.git
cd langextract
pip install -e .
```

## API Key Setup

Set your API key for cloud-hosted models (e.g., Gemini, OpenAI).

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

```bash
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
EOF
echo '.env' >> .gitignore
```

**Option 3: Direct API Key (Testing Only)**

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"
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
        "location": "global"
    }
)
```

## Adding Custom Model Providers

Extend LangExtract's capabilities with custom LLM providers using a plugin system. See the [Provider System Documentation](langextract/providers/README.md).

## Examples

### *Romeo and Juliet* Full Text Extraction

Extract structured data from the full text of *Romeo and Juliet* (Project Gutenberg).

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

Extract medical information from clinical text.

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

An interactive demo on HuggingFace Spaces for structuring radiology reports.

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Explore and contribute to community-built plugins.  See the [Community Provider Plugins](COMMUNITY_PROVIDERS.md).

## Contributing

Contributions are welcome!  See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines.

## Testing

```bash
pip install -e ".[test]"
pytest tests
```

```bash
tox  # runs pylint + pytest
```

```bash
tox -e ollama-integration  # Ollama integration test
```

## Development

### Code Formatting

```bash
./autoformat.sh
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

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, usage is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**