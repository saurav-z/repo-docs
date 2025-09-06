<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text

**Unlock valuable insights from text documents with LangExtract, a powerful Python library leveraging Large Language Models (LLMs) for structured information extraction.**

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.17015089.svg)](https://doi.org/10.5281/zenodo.17015089)

## Key Features

*   **Precise Source Grounding:** Every extraction is linked to its original text location for easy verification.
*   **Reliable Structured Outputs:** Consistent output schemas ensure robust, structured results.
*   **Optimized for Long Documents:** Handles large documents with chunking and parallel processing for high recall.
*   **Interactive Visualization:** Generate interactive HTML files to visualize and review extractions in context.
*   **Flexible LLM Support:** Supports cloud-based (Gemini, OpenAI) and local (Ollama) LLMs.
*   **Domain Agnostic:** Define extraction tasks for any domain with just a few examples.
*   **Leverages LLM Knowledge:** Utilize prompt engineering and few-shot examples for informed extraction.

## Getting Started

### Installation

Install LangExtract using pip:

```bash
pip install langextract
```

### Quick Example

Extract structured information with just a few lines of code:

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

# The input text to be processed
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

# Run the extraction
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

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

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

**Important**:  Using cloud-hosted models like Gemini requires an API key. See [API Key Setup](#api-key-setup-for-cloud-models) for details.

### Scaling to Longer Documents

Process entire documents directly from URLs:

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

**[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

## API Key Setup for Cloud Models

When using cloud-hosted models (like Gemini or OpenAI), you'll need to set up an API key.

*   Get API keys from:
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

LangExtract supports local inference using Ollama:

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

For detailed installation, Docker setup, and examples, see [`examples/ollama/`](examples/ollama/).

## Adding Custom Model Providers

LangExtract offers a plugin system for custom LLM providers, letting you extend functionality without modifying the core code. See [Custom Provider Plugin Example](examples/custom_provider_plugin/) for more information.

## More Examples

Explore more use cases and examples:

*   ***Romeo and Juliet* Full Text Extraction**: Demonstrates processing entire documents directly from URLs. **[View Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**
*   **Medication Extraction**: Illustrates structured medical information extraction from clinical text. **[View Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**
*   **Radiology Report Structuring: RadExtract**: Interactive demo on HuggingFace Spaces. **[View Demo →](https://huggingface.co/spaces/google/radextract)**

## Community Providers

Discover community-created providers or add your own! Check out our [Community Provider Plugins](COMMUNITY_PROVIDERS.md) registry.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to get started. You must sign a [Contributor License Agreement](https://cla.developers.google.com/about) before submitting patches.

## Testing

Run tests from source:

```bash
# Clone the repository
git clone https://github.com/google/langextract.git
cd langextract

# Install with test dependencies
pip install -e ".[test]"

# Run all tests
pytest tests
```

## Development

See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for full development guidelines.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).  For health-related applications, use of LangExtract is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Get started with LangExtract today and unlock the power of structured data extraction!**  For more details, visit the [LangExtract GitHub repository](https://github.com/google/langextract).