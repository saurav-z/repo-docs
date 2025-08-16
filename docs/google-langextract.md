<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Information from Text with LLMs

**Unlock the power of Large Language Models to extract and organize key information from unstructured text, making data analysis and knowledge discovery easier than ever.**

LangExtract, a powerful Python library, empowers you to transform unstructured text data into structured insights using the latest LLMs.  Leverage your preferred models, including cloud-based (Gemini, OpenAI) and local (Ollama) options, to extract crucial details, relationships, and more from diverse documents.

## Key Features

*   ✅ **Precise Source Grounding:**  Each extracted element is linked directly to its source text, enabling easy verification and highlighting.
*   ✅ **Reliable Structured Outputs:**  Ensure consistent, structured results with a schema based on your few-shot examples.
*   ✅ **Optimized for Long Documents:**  Handles large texts efficiently through chunking, parallel processing, and multi-pass extraction.
*   ✅ **Interactive Visualization:**  Generate self-contained HTML visualizations to explore and review extracted entities in context.
*   ✅ **Flexible LLM Support:** Works with cloud models (Gemini, OpenAI) and local LLMs via Ollama, providing model choice.
*   ✅ **Domain Agnostic:** Define extraction tasks for any domain using just a few examples; no model fine-tuning required.
*   ✅ **Leverages LLM World Knowledge:** Utilize LLM knowledge by using precise prompt wording and example attributes.

## Quick Start

**[Visit the LangExtract GitHub Repository](https://github.com/google/langextract) to get started.**

### 1. Define Your Extraction Task

Craft a prompt that outlines your desired extraction task, and provide a high-quality example to guide the model.

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

Use the `lx.extract` function with your text, prompt, and example.

```python
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```

> **Model Selection**: `gemini-2.5-flash` is the recommended default. For complex tasks use `gemini-2.5-pro`.  For production, use Tier 2 Gemini.

### 3. Visualize the Results

Save extractions as a `.jsonl` file and generate an interactive HTML visualization.

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

### Scaling to Longer Documents

Process lengthy texts efficiently using URLs, parallel processing, and sequential passes.

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

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

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

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup

### Cloud Models (Gemini, OpenAI)

Obtain your API keys and set them via:

*   **AI Studio:**  [https://aistudio.google.com/app/apikey](https://aistudio.google.com/app/apikey) (Gemini)
*   **Vertex AI:** [https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview](https://cloud.google.com/vertex-ai/generative-ai/docs/sdks/overview) (for enterprise use)
*   **OpenAI Platform:** [https://platform.openai.com/api-keys](https://platform.openai.com/api-keys) (OpenAI)

**Set API keys using:**

*   **Environment Variable:** `export LANGEXTRACT_API_KEY="your-api-key-here"`
*   **.env File:** (Recommended) Create a `.env` file with `LANGEXTRACT_API_KEY=your-api-key-here` and add `.env` to `.gitignore`.
*   **Directly in Code:** (Not recommended for production)

## Adding Custom Model Providers

Extend LangExtract by adding support for new LLM providers with a plugin system. See [Provider System Documentation](langextract/providers/README.md).

## Using OpenAI Models

Install OpenAI dependencies and use `model_id="gpt-4o"` and set `api_key`. Requires `fence_output=True` and `use_schema_constraints=False`.

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
    model_id="gpt-4o",  # Automatically selects OpenAI provider
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)
```

## Using Local LLMs with Ollama

Use local LLMs with Ollama without requiring API keys.

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

Install Ollama: [ollama.com](https://ollama.com/), then pull the desired model, and run `ollama serve`.

## More Examples

### *Romeo and Juliet* Full Text Extraction

**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction

**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract

**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md).

## Testing

Run tests locally from source, with test dependencies, or with `tox`.

## Development

*   Automated code formatting with `./autoformat.sh` and other tools.
*   Pre-commit hooks for automatic formatting checks.
*   Linting before submitting PRs.

## Disclaimer

This is not an officially supported Google product. Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and for health-related applications, the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).

---

**Happy Extracting!**