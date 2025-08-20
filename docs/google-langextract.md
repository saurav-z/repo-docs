<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Unlock Structured Data from Unstructured Text with LLMs

**Effortlessly transform unstructured text into structured data using the power of Large Language Models (LLMs) with LangExtract.**

LangExtract is a versatile Python library designed to extract structured information from unstructured text documents. It leverages LLMs to identify and organize key details based on user-defined instructions, making it ideal for processing clinical notes, reports, and other text-based content. [Explore the LangExtract repository](https://github.com/google/langextract) for detailed information.

## Key Features:

*   ✅ **Precise Source Grounding:** Every extraction is linked to its original text location for easy verification.
*   ✅ **Consistent Structured Outputs:** Enforces a predefined schema for reliable results, leveraging model capabilities for Gemini.
*   ✅ **Optimized for Long Documents:** Efficiently handles large documents using chunking, parallel processing, and multiple passes.
*   ✅ **Interactive Visualization:** Generates interactive HTML files to visualize and review extracted entities in context.
*   ✅ **Flexible LLM Support:** Supports various LLMs, including Google Gemini (cloud-based) and local open-source models via Ollama.
*   ✅ **Domain Agnostic:** Adaptable to any domain with just a few examples; no model fine-tuning required.
*   ✅ **Leverages LLM Knowledge:** Utilize precise prompting and examples to influence LLM knowledge utilization for tailored extractions.

## Quick Start

LangExtract allows you to extract structured information with just a few lines of code.

```python
import langextract as lx
import textwrap

# 1. Define the extraction task with a prompt and extraction rules
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

# Save and visualize results
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

**Model Selection**: `gemini-2.5-flash` is the recommended default, offering an excellent balance of speed, cost, and quality. For highly complex tasks requiring deeper reasoning, `gemini-2.5-pro` may provide superior results.

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

## API Key Setup for Cloud Models

Set up API keys for Gemini or OpenAI models using environment variables or `.env` files.  On-device models don't require API keys.

*   Get API keys from [AI Studio](https://aistudio.google.com/app/apikey) for Gemini models.
*   Get API keys from [OpenAI Platform](https://platform.openai.com/api-keys) for OpenAI models.

**Environment Variable:**

```bash
export LANGEXTRACT_API_KEY="your-api-key-here"
```

**`.env` file (Recommended):**

```bash
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
EOF
echo '.env' >> .gitignore
```

## Adding Custom Model Providers

LangExtract supports a plugin system for adding custom LLM providers, allowing you to extend functionality without changing core code. See [Provider System Documentation](langextract/providers/README.md) for more details.

## Using OpenAI Models

Install the OpenAI dependency and use with your OpenAI API key.

```bash
pip install langextract[openai]
```

```python
import langextract as lx
import os

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

## Using Local LLMs with Ollama

Run local inference with Ollama. Install Ollama from [ollama.com](https://ollama.com/), and then run the following commands:

```bash
ollama pull gemma2:2b
ollama serve
```

```python
import langextract as lx

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemma2:2b",
    model_url="http://localhost:11434",
    fence_output=False,
    use_schema_constraints=False
)
```

## More Examples

*   ***Romeo and Juliet*** Full Text Extraction: Demonstrates extraction from the complete play using parallel processing. **[View Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**
*   Medication Extraction: Illustrates extracting structured medical information. **[View Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**
*   Radiology Report Structuring: See a live demo of RadExtract on Hugging Face Spaces. **[View Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome; see [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines. You must sign a [Contributor License Agreement](https://cla.developers.google.com/about).

## Testing

Run tests locally from the source:

```bash
pip install -e ".[test]"
pytest tests
```

## Development

Use auto-formatting, pre-commit hooks, and linting for consistent code style.

## Disclaimer

This is not an official Google product. See the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE).  For health-related applications, usage is subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).