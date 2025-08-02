# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, structured JSON data with Instructor, built on Pydantic for validation, type safety, and seamless integration.**  [See the original repo](https://github.com/567-labs/instructor) for more information.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Eliminate the need for complex JSON schema definitions and manual parsing.
*   **Automatic Validation:** Leverage Pydantic for robust data validation, ensuring data integrity.
*   **Provider Agnostic:** Works seamlessly with leading LLM providers, including OpenAI, Anthropic, Google, and local models like Ollama.
*   **Automatic Retries:** Handle LLM inconsistencies with built-in retry mechanisms.
*   **Streaming Support:** Process partial object outputs in real-time for improved user experience.
*   **Nested Object Support:** Easily extract and manage complex, hierarchical data structures.

## The Problem: Structured Data Extraction Challenges

Extracting structured data from LLMs is a complex process that often involves:

1.  Writing intricate JSON schemas.
2.  Dealing with validation errors.
3.  Implementing retry logic for failed extractions.
4.  Parsing unstructured responses.
5.  Managing various provider API differences.

## The Solution: Instructor Simplifies Everything

Instructor streamlines the entire process, offering a simple and efficient interface:

| **Challenge**      | **Traditional Approach**                                                                                                                                                                                                                                     | **Instructor Solution**                                                                                                                                                                                             |
| :------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| JSON Schema Creation | Requires manual creation and maintenance of complex JSON schemas.                                                                                                                                                                                             | Uses Pydantic models. Just define your Python model.                                                                                                                                                            |
| Validation           | Requires manual validation of the extracted JSON data, including error handling.                                                                                                                                                                          | Built-in Pydantic validation ensures data correctness.                                                                                                                                                            |
| Retries              | Requires custom logic to handle extraction failures and retries.                                                                                                                                                                                              | Automatic retries are handled when validation fails.                                                                                                                                                              |
| Provider Management | Requires understanding of different provider APIs and custom parsing logic for each provider                                                                                                                                                                   | Instructor uses the same API regardless of the provider.                                                                                                                                                        |
| Code overhead      | Significant code is required for parsing, validating, and handling extraction failures.                                                                                                                                                                     | Reduces the amount of code needed for extraction, making your code cleaner and easier to read.                                                                                                              |

## Installation

Get started in seconds:

```bash
pip install instructor
```

Or with your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## LLM Provider Compatibility

Instructor supports a wide range of LLM providers, using the same simple API:

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

## Powerful Production-Ready Features

### Automatic Retries

Automate retries based on validation failures:

```python
from pydantic import BaseModel, field_validator


class User(BaseModel):
    name: str
    age: int

    @field_validator('age')
    def validate_age(cls, v):
        if v < 0:
            raise ValueError('Age must be positive')
        return v


# Instructor automatically retries when validation fails
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,
)
```

### Streaming Support

Process data as it's generated:

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,
):
    print(partial_user)
    # User(name=None, age=None)
    # User(name="John", age=None)
    # User(name="John", age=25)
```

### Nested Objects

Easily handle complex data structures:

```python
from typing import List


class Address(BaseModel):
    street: str
    city: str
    country: str


class User(BaseModel):
    name: str
    age: int
    addresses: List[Address]


# Instructor handles nested objects automatically
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Adopted by the community

Instructor is trusted by a wide range of developers and companies in production:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started Today

### Simple Extraction

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

### Multi-Language Support

Instructor's simple API is available in many languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Instructor?

**Instructor vs. Alternatives**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing or custom solutions required.

## Contributing

Contributions are welcome! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>