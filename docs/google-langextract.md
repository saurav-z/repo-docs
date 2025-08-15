<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Unlock Structured Insights from Unstructured Text

**Transform raw text into actionable data with LangExtract, a powerful Python library leveraging Large Language Models (LLMs) to extract structured information from your documents.** ([View on GitHub](https://github.com/google/langextract))

[![PyPI version](https://img.shields.io/pypi/v/langextract.svg)](https://pypi.org/project/langextract/)
[![GitHub stars](https://img.shields.io/github/stars/google/langextract.svg?style=social&label=Star)](https://github.com/google/langextract)
![Tests](https://github.com/google/langextract/actions/workflows/ci.yaml/badge.svg)

## Key Features:

*   ✅ **Precise Source Grounding:** Trace extractions back to their original text location with easy highlighting.
*   ✅ **Consistent Structured Outputs:** Enforce a defined output schema for reliable, structured results using few-shot examples.
*   ✅ **Optimized for Long Documents:** Efficiently process large texts with chunking, parallel processing, and multiple passes for high recall.
*   ✅ **Interactive Visualization:** Generate self-contained HTML files to explore extracted entities in their original context.
*   ✅ **Flexible LLM Support:** Works seamlessly with cloud-based LLMs (Gemini, OpenAI) and local open-source models via Ollama.
*   ✅ **Domain Agnostic:** Easily adapt to any domain with custom extraction tasks defined by a few examples.
*   ✅ **Leverages LLM Knowledge:** Utilize prompt wording and few-shot examples to influence how the extraction task may utilize LLM knowledge. The accuracy of any inferred information and its adherence to the task specification are contingent upon the selected LLM, the complexity of the task, the clarity of the prompt instructions, and the nature of the prompt examples.

## Quick Start

Get started extracting structured data in minutes!

### 1. Define Your Extraction Task

Create a prompt describing what you want to extract and guide the model with examples.

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

Pass your input text, prompt, and examples to `lx.extract`.

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

### 3. Visualize the Results

Save your extractions to a `.jsonl` file and generate an interactive HTML visualization.

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

This creates an animated and interactive HTML file:

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

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

Set up an API key for cloud-hosted models (Gemini, OpenAI). On-device models like Ollama do not require an API key.

*   **Option 1: Environment Variable**
    ```bash
    export LANGEXTRACT_API_KEY="your-api-key-here"
    ```
*   **Option 2: .env File (Recommended)**

    Create a `.env` file in the same directory as your script.

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
*   **Option 3: Direct API Key (Not Recommended for Production)**

    ```python
    result = lx.extract(
        text_or_documents=input_text,
        prompt_description="Extract information...",
        examples=[...],
        model_id="gemini-2.5-flash",
        api_key="your-api-key-here"  # Only use this for testing/development
    )
    ```

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
**Quick setup:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.

## More Examples

*   **Full Text Extraction of *Romeo and Juliet***: Explore detailed results and performance insights from a full novel extraction. **[View Example](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**
*   **Medication Extraction**: (For illustrative purposes only) See how LangExtract can extract structured medical information. **[View Examples](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**
*   **RadExtract Demo**: Try a live interactive demo for radiology report structuring on HuggingFace Spaces. **[View Demo](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines.

## Testing

Run tests locally:
```bash
pip install -e ".[test]"
pytest tests
```
or run the full CI matrix locally with tox:
```bash
tox
```

## Development
See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for full development guidelines, including code formatting and linting instructions.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, use is also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).