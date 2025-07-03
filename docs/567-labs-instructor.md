# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your AI workflows and reliably extract structured JSON data from any Language Model with Instructor, built on Pydantic.** Learn more at the original repository: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

Instructor makes it easy to reliably extract structured data from any LLM, eliminating the need for complex JSON parsing, error handling, and retries. Use the power of Pydantic for validation, type safety, and IDE support.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Define your data model with Pydantic and get structured data directly.
*   **Automatic Validation:** Ensures data integrity and type safety.
*   **Intelligent Retries:** Automatically retries failed extractions based on validation errors.
*   **Streaming Support:** Stream partial objects as they're generated, improving user experience and efficiency.
*   **Nested Object Support:** Handles complex, nested data structures effortlessly.
*   **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, and more) using a unified API.

## How Instructor Simplifies LLM Data Extraction

Instructor dramatically reduces the complexity of extracting structured data compared to traditional methods:

| Without Instructor                                                                   | With Instructor                                                                      |
| :----------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------- |
| Requires manual JSON schema creation, parsing, and validation.                       | Define a Pydantic model and let Instructor handle the rest.                           |
| Demands error handling and retry mechanisms to manage failed extractions.             | Instructor automatically retries on validation errors.                              |
| Time-consuming, complex, and prone to errors, particularly with different providers. | Uses a simple, unified API across all providers, saving development time.           |

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Production-Ready Features

Instructor includes powerful features for production environments:

*   **Automatic Retries:** Handles validation failures automatically with error messages.
*   **Streaming Support:** Stream partial objects as they are generated, providing real-time updates.
*   **Nested Objects:** Supports complex, nested data structures.

## Getting Started

### Basic Extraction Example

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

## Used in Production By

Join the community of 100,000+ developers and companies that trust Instructor to build AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Instructor is used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Explore Instructor in Multiple Languages

Instructor's simple API is available in many languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

## Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides.
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes.
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices.
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community.

## Why Choose Instructor?

*   **vs Raw JSON mode:** Instructor offers automatic validation, retries, streaming, and nested object support. It eliminates manual schema creation.
*   **vs LangChain/LlamaIndex:** Instructor focuses on structured extraction, providing a lighter, faster, and more debuggable solution.
*   **vs Custom solutions:** Benefit from a battle-tested solution used by thousands of developers, handling edge cases effectively.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>