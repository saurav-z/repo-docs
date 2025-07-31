# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and build reliable AI applications by extracting structured data like JSON with ease.**  [View the project on GitHub](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Reliable Data Extraction:** Get consistent, validated JSON output from any LLM.
*   **Pydantic Integration:** Leverage Pydantic for type safety, validation, and seamless IDE support.
*   **Automatic Retries:** Handle extraction failures gracefully with built-in retry mechanisms.
*   **Streaming Support:**  Receive partial objects in real-time as they are generated.
*   **Nested Object Support:** Effortlessly extract complex, nested data structures.
*   **Cross-Provider Compatibility:** Works seamlessly with all major LLM providers.
*   **Production-Ready:** Trusted by over 100,000 developers and companies.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Choose Instructor?

Instructor dramatically simplifies the process of extracting structured data from LLMs. Here's how it compares to traditional approaches:

| **Challenge**             | **Traditional Approach**                                                                                                                                                                                     | **Instructor**                                                                                                                                    |
| ------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Schema Definition**     | Requires manual JSON schema creation.                                                                                                                                                                           | Uses Pydantic models for simple and elegant schema definition.                                                                                  |
| **Validation**            | Manual validation of extracted data required.                                                                                                                                                                | Automatic validation based on Pydantic models, ensuring data integrity.                                                                              |
| **Error Handling**        | Needs complex logic to handle extraction errors and retries.                                                                                                                                                    | Built-in retry mechanisms for handling extraction failures.                                                                                     |
| **Data Parsing**          | Requires manual parsing of unstructured LLM responses.                                                                                                                                                        | Parses unstructured responses automatically and converts them into structured data.                                                                   |
| **Provider API Variation** | Requires adapting to different API formats from various providers.                                                                                                                                        | Provides a unified API that works consistently across all major LLM providers (OpenAI, Anthropic, Google, Ollama, and more).                    |

## Get Started

### Installation

Install Instructor using pip:

```bash
pip install instructor
```

Or with your package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Example

Easily extract structured data from text using Instructor:

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

*   **Automatic Retries:**  Automatically retry failed extractions.
*   **Streaming Support:**  Stream partial objects as they're generated.
*   **Nested Objects:** Extract complex, nested data structures effortlessly.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

### Provider Compatibility

Instructor works seamlessly with a wide range of LLM providers:

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

# All use the same API!
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Used in Production By

Trusted by over 100,000 developers and companies building AI applications:

-   **3M+ monthly downloads**
-   **10K+ GitHub stars**
-   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>