<p align="center">
  <a href="https://github.com/google/langextract">
    <img src="https://raw.githubusercontent.com/google/langextract/main/docs/_static/logo.svg" alt="LangExtract Logo" width="128" />
  </a>
</p>

# LangExtract: Effortlessly Extract Structured Data from Unstructured Text Using LLMs

**LangExtract** is a powerful Python library that empowers you to transform unstructured text into structured, actionable data using the capabilities of Large Language Models (LLMs).  [Explore the LangExtract GitHub repository](https://github.com/google/langextract) to get started.

**Key Features:**

*   ✅ **Precise Source Grounding:**  Pinpoints extractions to the exact location in the original text for easy verification.
*   ✅ **Robust Structured Outputs:** Ensures consistent, schema-based outputs leveraging LLM generation.
*   ✅ **Optimized for Long Documents:** Efficiently handles large documents with chunking, parallel processing, and multi-pass extraction.
*   ✅ **Interactive Visualization:** Generates interactive HTML visualizations to review and analyze extracted entities in context.
*   ✅ **Flexible LLM Support:** Works with a variety of LLMs, including Gemini, OpenAI, and local open-source models via Ollama.
*   ✅ **Domain Agnostic:** Adaptable to any domain; define extraction tasks with just a few examples without model fine-tuning.
*   ✅ **Leverages LLM World Knowledge:** Utilize precise prompts and few-shot examples to draw upon the LLM's world knowledge.

## Core Functionality

LangExtract simplifies the extraction process with a few lines of code.

1.  **Define Your Extraction Task:** Create a prompt to describe what you want to extract.
2.  **Provide Guidance:** Give a few examples to guide the model.
3.  **Run Extraction:**  Use the `lx.extract` function, specifying the input text, prompt, examples, and desired model.
4.  **Visualize Results:** Save results to a `.jsonl` file and generate an interactive HTML visualization to review the extracted data.

**Example:**

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

input_text = "Lady Juliet gazed longingly at the stars, her heart aching for Romeo"

result = lx.extract(
    text_or_documents=input_text,
    prompt_description=prompt,
    examples=examples,
    model_id="gemini-2.5-flash",
)

lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
html_content = lx.visualize("extraction_results.jsonl")
with open("visualization.html", "w") as f:
    if hasattr(html_content, 'data'):
        f.write(html_content.data)  # For Jupyter/Colab
    else:
        f.write(html_content)
```

**Example Output:**

![Romeo and Juliet Basic Visualization ](https://raw.githubusercontent.com/google/langextract/main/docs/_static/romeo_juliet_basic.gif)

## Key Use Cases & Examples

*   **[Romeo and Juliet Full Text Extraction](https://github.com/google/langextract/blob/main/docs/examples/longer_text_example.md):**  Process entire documents directly from URLs, demonstrating efficient long document handling.
*   **[Medication Extraction](https://github.com/google/langextract/blob/main/docs/examples/medication_examples.md):** Extract structured medical information from clinical text.  _Note: This is for illustrative purposes only, and should not be used for medical advice._
*   **[Radiology Report Structuring: RadExtract](https://huggingface.co/spaces/google/radextract):** See a live demo that automatically structures radiology reports.

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

Cloud-hosted models (Gemini, OpenAI) require an API key. Local LLMs (Ollama) do not.

**Recommended:** Add your API key to a `.env` file:

```bash
# Add API key to .env file
cat >> .env << 'EOF'
LANGEXTRACT_API_KEY=your-api-key-here
EOF

# Keep your API key secure
echo '.env' >> .gitignore
```

or,

**Other Options:**
*   Set an environment variable: `export LANGEXTRACT_API_KEY="your-api-key-here"`
*   Provide the API key directly in your code (not recommended for production).

## Model Support

*   **Gemini:** Use `model_id="gemini-2.5-flash"` (recommended) or `gemini-2.5-pro`.
*   **OpenAI:** Install `pip install langextract[openai]` and use `model_id="gpt-4o"`
*   **Ollama:** Install Ollama from [ollama.com](https://ollama.com/), run `ollama pull gemma2:2b`, then `ollama serve`. Use `model_id="gemma2:2b"` and specify `model_url="http://localhost:11434"` in your `lx.extract` call.

## Contributing

Contributions are welcome! See [CONTRIBUTING.md](https://github.com/google/langextract/blob/main/CONTRIBUTING.md) for guidelines.

## Testing

```bash
pip install -e ".[test]"
pytest tests
```

## Disclaimer

This is not an officially supported Google product.  Use is subject to the [Apache 2.0 License](https://github.com/google/langextract/blob/main/LICENSE). For health-related applications, also subject to the [Health AI Developer Foundations Terms of Use](https://developers.google.com/health-ai-developer-foundations/terms).