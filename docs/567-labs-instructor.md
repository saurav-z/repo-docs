# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions with Instructor, the Python library that effortlessly extracts structured data, validates responses, and handles retries – all with a simple and intuitive interface.**  Check out the original repository on GitHub: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Eliminate complex JSON schema writing and manual parsing.
*   **Built on Pydantic:** Leverage Pydantic for type safety, validation, and IDE support.
*   **Automatic Retries:** Handle failed extractions gracefully with built-in retry mechanisms.
*   **Streaming Support:** Stream partial objects as they're generated for improved responsiveness.
*   **Nested Object Support:** Effortlessly extract complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with a wide range of LLM providers, including OpenAI, Anthropic, Google, and local models.
*   **Production Ready:** Used by 100,000+ developers in production, trusted by OpenAI, Google, Microsoft, AWS, and many YC startups

## The Problem Instructor Solves

Getting reliable structured data from LLMs can be a tedious process. Instructor simplifies this.

**Here's the difference:**

| **Without Instructor**                                                                                | **With Instructor**                                                                                                  |
| :----------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| Complex JSON Schemas, Manual Validation, Error Handling, Retries, Provider Specific API interactions | Simple Model Definition, Automatic Validation, Retries, Nested Objects, Supports Multiple Providers, and Streaming |

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

## Provider Support

Use the same code across all major LLM providers.

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

## Production-Ready Features in Detail

### Automatic Retries
Instructor automatically retries on validation errors.

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

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,
)
```

### Streaming Support

Get partial objects as they're generated:

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,
):
    print(partial_user)
```

### Nested Objects
Extract complex data structures with ease:

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

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Quickstart: Basic Extraction

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

## Multiple Languages

Instructor's API is available in multiple languages:

-   [Python](https://python.useinstructor.com) - The original
-   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
-   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
-   [Go](https://go.useinstructor.com) - Go implementation
-   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
-   [Rust](https://rust.useinstructor.com) - Rust implementation

## Resources for Further Learning

-   [Documentation](https://python.useinstructor.com) - Comprehensive guides
-   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
-   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
-   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Instructor?

*   **Simplified workflow**: No need for complex JSON schema writing or manual parsing.
*   **Robustness**: Automatic validation, retries, and nested object support handle various edge cases.
*   **Efficiency**: Lighter, faster, and easier to debug than alternatives like LangChain or LlamaIndex.
*   **Production Proven**: Leveraged by thousands of developers.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>