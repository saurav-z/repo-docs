<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Extract Structured Information from Unstructured Text with LLMs

**LangExtract empowers you to effortlessly extract structured data from text documents using Large Language Models, turning raw text into valuable insights.**  Visit the [original repo](https://github.com/google/langextract) for the source code and more details.

## Key Features

*   ✅ **Precise Source Grounding:**  Pinpoints extractions to their exact location in the source text.
*   ✅ **Reliable Structured Outputs:** Ensures consistent, structured data with few-shot example-driven schemas.
*   ✅ **Optimized for Long Documents:**  Handles large documents with chunking, parallel processing, and multiple passes for high recall.
*   ✅ **Interactive Visualization:** Generates self-contained, interactive HTML files for instant review.
*   ✅ **Flexible LLM Support:** Works with your preferred models: Google Gemini (cloud-based) and local open-source models via Ollama.
*   ✅ **Domain Agnostic:** Adaptable to any domain with custom extraction tasks defined via examples, without model fine-tuning.
*   ✅ **Leverages LLM Knowledge:** Utilizes LLM world knowledge through prompt wording and few-shot examples (accuracy depends on model and task complexity).

## Quick Start

### 1. Define Your Extraction Task
   *   Create a prompt to describe what you want to extract.
   *   Provide a high-quality example to guide the model.

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

```python
input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"
result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)
```
> **Model Selection:** `gemini-2.5-flash` (recommended default). Consider `gemini-2.5-pro` for complex tasks. Tier 2 Gemini quota suggested for large-scale use to avoid rate limits.  See the [rate-limit documentation](https://ai.google.dev/gemini-api/docs/rate-limits#tier-2) for details.  Consult [official model version documentation](https://cloud.google.com/vertex-ai/generative-ai/docs/learn/model-versions) to stay informed.

### 3. Visualize the Results

```python
lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```
![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

### Scaling to Longer Documents
*  Process documents directly from URLs with parallel processing and enhanced sensitivity.

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
**[See the full *Romeo and Juliet* extraction example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

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

```bash
git clone https://github.com/google/langextract.git
cd langextract
pip install -e .  # For basic install
# pip install -e ".[dev]" #For development (includes linting tools)
# pip install -e ".[test]" #For testing (includes pytest)
```

### Docker

```bash
docker build -t langextract .
docker run --rm -e LANGEXTRACT_API_KEY="your-api-key" langextract python your_script.py
```

## API Key Setup for Cloud Models

*   Requires API keys for Gemini and OpenAI. On-device models do not require an API key. For local LLMs, LangExtract offers built-in support for Ollama.

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
echo '.env' >> .gitignore
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

**Option 3: Direct API Key (Not Recommended for Production)**

```python
result = lx.extract(
    text_or_documents=input_text,
    prompt_description="Extract information...",
    examples=[...],
    model_id="gemini-2.5-flash",
    api_key="your-api-key-here"  # Only use this for testing/development
)
```

## Adding Custom Model Providers

*   Extend LangExtract to support new LLMs via a plugin system.  Create and distribute providers independently as Python packages.

See detailed guide in [Provider System Documentation](langextract/providers/README.md).

## Using OpenAI Models

*   Requires optional dependency: `pip install langextract[openai]`

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
* Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`.
*   See [`examples/ollama/`](examples/ollama/) for detailed installation and examples.

## More Examples

### *Romeo and Juliet* Full Text Extraction
**[View *Romeo and Juliet* Full Text Example →](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md)**

### Medication Extraction
**[View Medication Examples →](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md)**

### Radiology Report Structuring: RadExtract
**[View RadExtract Demo →](https://huggingface.co/spaces/google/radextract)**

## Contributing

*   Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) to get started.
*   Sign a [Contributor License Agreement](https://cla.developers.google.com/about).

## Testing

*   Run tests locally:
    ```bash
    git clone https://github.com/google/langextract.git
    cd langextract
    pip install -e ".[test]"
    pytest tests
    ```
*   Or reproduce the full CI matrix locally with tox:
    ```bash
    tox
    ```
*   Ollama integration testing:
    ```bash
    tox -e ollama-integration
    ```

## Development

### Code Formatting

```bash
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

This is not an officially supported Google product. Use subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE) and, for health-related applications, the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).