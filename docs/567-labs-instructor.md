# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and reliably extract structured data with Instructor – the Python library built on Pydantic for seamless validation, type safety, and streamlined development.**  Learn more on the original GitHub repo: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Data Extraction:**  Eliminate the complexity of JSON parsing, error handling, and retries. Just define your data model and get structured data.
*   **Pydantic-Powered Validation:** Leverage the power of Pydantic for robust validation and type safety, ensuring data integrity.
*   **Automatic Retries:**  Instructor automatically retries failed extractions, handling common LLM issues.
*   **Streaming Support:** Stream partial objects as they're generated, for improved user experience.
*   **Nested Object Support:** Effortlessly extract complex, nested data structures with ease.
*   **Cross-Provider Compatibility:** Works with all major LLM providers, including OpenAI, Anthropic, Google, and local models.
*   **Production-Ready:** Built to handle real-world scenarios, with features like automatic retries and streaming.

## The Problem Instructor Solves

Getting reliable structured data from Large Language Models (LLMs) can be a complex and time-consuming process.  Instructor streamlines this process, eliminating the need for:

*   Writing complex JSON schemas
*   Handling validation errors manually
*   Implementing retry mechanisms
*   Parsing unstructured responses
*   Adapting to different provider APIs

## Get Started in Seconds

Install Instructor using pip:

```bash
pip install instructor
```

Or with your preferred package manager (uv, poetry):

```bash
uv add instructor
poetry add instructor
```

## Example: Basic Data Extraction

Effortlessly extract structured data from text:

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

## Works with Major LLM Providers

Use the same code with any LLM provider:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# With API keys directly
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

# All use the same API!
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-Ready Features

*   **Automatic Retries:** Instructor automatically retries failed validations, using error messages for enhanced accuracy.
*   **Streaming Support:** Stream partial objects as they're generated, improving responsiveness.
*   **Nested Objects:** Extract complex nested data structures automatically.

## Used in Production

Instructor is trusted by over 100,000 developers and companies for building AI applications.

*   3M+ monthly downloads
*   10K+ GitHub stars
*   1000+ community contributors

## Why use Instructor?

**vs Raw JSON mode:** Instructor automates validation, retries, streaming, and nested object support. No manual schema creation.

**vs LangChain/LlamaIndex:** Instructor offers a focused, lightweight, and efficient solution tailored specifically for structured extraction.

**vs Custom solutions:** Benefit from a battle-tested solution refined by thousands of developers, handling edge cases effectively.

## Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>