# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your interactions with large language models (LLMs) and get reliable, validated JSON outputs with Instructor.**

[Visit the original repo on GitHub](https://github.com/567-labs/instructor)

## Key Features

*   **Effortless Extraction:** Define your desired data structure using Pydantic models and let Instructor handle the rest.
*   **Automatic Validation:** Built-in Pydantic validation ensures data quality and type safety.
*   **Simplified Error Handling:**  Automated retries with error messages streamline extraction.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and local models (Ollama).
*   **Streaming Support:** Receive partial object results as they're generated.
*   **Nested Object Extraction:** Easily handle complex, nested data structures.

## Why Instructor?

Instructor provides a streamlined, robust solution for extracting structured data from LLMs, unlike raw JSON mode, LangChain/LlamaIndex, or custom solutions. It provides automatic validation, retries, streaming, and nested object support. No manual schema writing.

## Installation

Get started in seconds:

```bash
pip install instructor
```

## Getting Started

### Basic Extraction

Extract structured data with ease:

```python
from pydantic import BaseModel
import instructor

client = instructor.from_provider("openai/gpt-4o-mini")

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

## Production-Ready Features

*   **Automatic Retries:**  Instructor automatically retries failed validations.
*   **Streaming Support:** Stream partial objects as they're generated.
*   **Nested Objects:** Extract complex, nested data structures with ease.

## Works with Every Major Provider

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")
```

## Used in Production By

Instructor is trusted by over 100,000 developers and companies.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Resources

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue).

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>