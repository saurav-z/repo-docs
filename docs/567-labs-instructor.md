# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for type safety and ease of use.**

[Visit the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features:

*   **Seamless Data Extraction:**  Easily extract structured data from any LLM, eliminating the need for complex JSON parsing.
*   **Pydantic Integration:** Leverage Pydantic for robust validation, type safety, and IDE support, ensuring data quality.
*   **Automatic Retries:** Handle LLM response failures with built-in automatic retries, improving reliability.
*   **Streaming Support:**  Receive and process data incrementally with streaming capabilities for faster responses.
*   **Nested Object Support:**  Effortlessly extract complex, nested data structures with intuitive handling.
*   **Multi-Provider Compatibility:** Works with a wide range of LLM providers (OpenAI, Anthropic, Google, Ollama, and more) with a unified API.

## The Problem Instructor Solves

Working with LLMs for structured data can be challenging.  Instructor streamlines this process by:

*   Eliminating the need for manual JSON schema creation.
*   Automating the handling of validation errors.
*   Providing automatic retries for failed extractions.
*   Simplifying the parsing of unstructured responses.
*   Offering a consistent interface across various provider APIs.

## Installation

Get started in seconds:

```bash
pip install instructor
```

Or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Get Started: Basic Extraction Example

Quickly extract structured data using Instructor:

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

Instructor offers powerful features for production environments:

*   **Automatic Retries:** Built-in retry mechanisms handle validation failures.
*   **Streaming Support:** Stream partial objects as they're generated.
*   **Nested Objects:** Extract complex, nested data structures effortlessly.

## Provider Compatibility

Use the same code across various LLM providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# With API keys directly (no environment variables needed)
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")
```

## Why Choose Instructor?

Instructor offers significant advantages over alternative approaches:

*   **Compared to Raw JSON Mode:**  Provides automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.
*   **Compared to LangChain/LlamaIndex:**  Focuses specifically on structured extraction, offering a lighter, faster, and more debuggable solution.
*   **Compared to Custom Solutions:**  Leverages a battle-tested framework, handling edge cases efficiently.

## Used in Production By

Instructor is trusted by 100,000+ developers and companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**
*   **Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.**

## Resources

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

Contribute to the project! Check out the [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>