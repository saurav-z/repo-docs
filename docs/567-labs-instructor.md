# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM integrations and get reliable, validated JSON with Instructor, a powerful Python library built on Pydantic.**  Find the original repo [here](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your desired output with Pydantic models and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity with built-in validation using Pydantic.
*   **Intelligent Retries:** Automatically retries failed extractions based on validation errors.
*   **Streaming Support:** Get partial objects as they're generated for improved user experience.
*   **Nested Object Support:** Handles complex, nested data structures seamlessly.
*   **Provider Agnostic:** Works with OpenAI, Anthropic, Google, Ollama, and more!
*   **Production-Ready:** Built for reliability and scalability in real-world applications.

## Why Choose Instructor?

Instructor solves the common challenges of working with LLMs:

*   **Eliminates manual JSON parsing and schema creation.**
*   **Automates error handling and retries.**
*   **Supports various LLM providers with a unified API.**
*   **Streamlines the development of robust, data-driven applications.**

## Getting Started

### Installation

Install Instructor in seconds:

```bash
pip install instructor
```

Or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Usage

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

*   **Automatic retries** to handle validation failures.
*   **Streaming support** to get partial objects as they're generated.
*   **Nested objects** for complex data structures.

## Supported Providers

Instructor supports a wide range of LLM providers:

*   OpenAI
*   Anthropic
*   Google
*   Ollama (Local)
*   And many more!

## Used by Leading Companies

Trusted by over 100,000 developers and companies, including teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Instructor Over Alternatives?

*   **vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.
*   **vs LangChain/LlamaIndex:** Instructor is laser-focused on structured extraction, making it lighter, faster, and easier to debug.
*   **vs Custom solutions:** Instructor is battle-tested and handles edge cases, saving you time and effort.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>