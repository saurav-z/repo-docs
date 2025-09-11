# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for type safety and ease of use.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your desired output model using Pydantic and get structured data effortlessly.
*   **Automatic Validation:** Ensures data integrity with built-in validation and error handling, reducing the need for manual schema creation and parsing.
*   **Automatic Retries:** Intelligently retries failed extractions based on validation errors, increasing reliability.
*   **Provider Agnostic:** Works seamlessly with all major LLM providers like OpenAI, Anthropic, Google, and more.
*   **Streaming Support:** Get partial objects as they are generated for enhanced responsiveness.
*   **Nested Object Support:** Handles complex, nested data structures with ease.

## The Problem Instructor Solves

Getting structured data from LLMs can be a complex process. Traditional approaches involve:

*   Writing intricate JSON schemas.
*   Manually handling validation errors.
*   Implementing retry mechanisms for failed extractions.
*   Parsing unstructured responses.
*   Adapting to different provider APIs.

**Instructor simplifies the process dramatically by:**

*   Eliminating the need for complex JSON schemas.
*   Automating validation and error handling.
*   Providing built-in retry mechanisms.
*   Offering a unified interface for various providers.

## Installation

Get started in seconds:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Examples

### Basic Extraction

Extract structured data from any text:

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

### Production-Ready Features

Instructor offers robust features for production environments:

*   **Automatic retries** for failed validations based on error messages.
*   **Streaming support** to receive partial objects as they're generated.
*   **Nested object support** to extract complex data structures.

### Integrations

Instructor's simple API is available in many languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

## Why Use Instructor?

**vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.

**vs LangChain/LlamaIndex**: Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.

**vs Custom solutions**: Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Used in Production

Trusted by over 100,000 developers and companies building AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

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