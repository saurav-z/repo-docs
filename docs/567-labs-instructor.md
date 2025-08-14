# Instructor: Effortlessly Extract Structured Data from LLMs

**Unlock the power of structured data with Instructor, a Python library that simplifies extracting reliable JSON from any Large Language Model (LLM).**  Visit the original repository for more details: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

Instructor leverages Pydantic for robust validation, type safety, and superior IDE support, making it easy to integrate structured data extraction into your LLM applications.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your data model with Pydantic and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity with built-in validation based on your Pydantic models.
*   **Intelligent Retries:** Automatically retries failed extractions, handling LLM inconsistencies.
*   **Streaming Support:**  Receive partial data as it's generated for real-time applications.
*   **Nested Object Support:**  Effortlessly handle complex, nested data structures.
*   **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, etc.).
*   **Production-Ready:** Trusted by 100,000+ developers and used in production environments.
*   **Multiple Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Choose Instructor?

Instructor simplifies the process of extracting structured data from LLMs, eliminating the need for:

*   Complex JSON schema creation
*   Manual validation and error handling
*   Retry logic implementation
*   Unstructured response parsing
*   Managing different provider APIs

Instructor provides a streamlined approach, reducing development time and increasing reliability.

### Installation

Install Instructor with pip:

```bash
pip install instructor
```

## Getting Started

### Basic Extraction

Easily extract structured data from any text:

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

### Advanced Features

*   **Automatic Retries:** Instructor automatically retries failed validations.
*   **Streaming:** Stream partial objects as they're generated.
*   **Nested Objects:** Extract complex, nested data structures.

## Used in Production by

Instructor is trusted by a vast community of developers and companies, including teams from OpenAI, Google, Microsoft, AWS, and various YC startups.  Join the 3M+ monthly downloads, 10K+ GitHub stars, and 1000+ community contributors.

## Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Comparison with Alternatives

*   **vs Raw JSON mode:** Instructor offers automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.
*   **vs LangChain/LlamaIndex:** Instructor focuses solely on structured extraction, providing a lighter, faster, and easier-to-debug solution.
*   **vs Custom solutions:** Instructor provides a battle-tested solution proven by thousands of developers, handling edge cases effectively.

## Contributing

We welcome contributions!  Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get involved.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>