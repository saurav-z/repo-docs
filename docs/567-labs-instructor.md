# Instructor: Effortlessly Extract Structured Data from LLMs

**Stop wrestling with LLM output and instantly get reliable, validated JSON with Instructor, the Python library that makes structured data extraction simple and efficient.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)
![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)
![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)
![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)

## Key Features

*   **Simplified Extraction:** Define your data model using Pydantic and let Instructor handle the rest.
*   **Automatic Validation:**  Ensures data integrity with built-in Pydantic validation.
*   **Intelligent Retries:** Automatically retries extractions when validation fails, improving reliability.
*   **Streaming Support:**  Get partial objects in real-time as the LLM generates them.
*   **Nested Object Support:** Effortlessly handle complex, nested data structures.
*   **Cross-Provider Compatibility:** Works seamlessly with leading LLM providers like OpenAI, Anthropic, Google, and local models (Ollama).
*   **Production-Ready:** Trusted by 100,000+ developers and companies in production.

## Why Instructor? The Problem Instructor Solves

Extracting structured data from LLMs is complex and error-prone.  Instructor simplifies the process by eliminating the need for manual JSON parsing, validation, and retry logic.

| **The Old Way (Without Instructor)**                                                                                                | **The Instructor Way**                                                                                                             |
| :------------------------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------------- |
| <ul><li>Requires writing complex JSON schemas.</li><li>Manual error handling and validation.</li><li>Need to implement retry mechanisms.</li><li>Parse unstructured responses and deal with different provider APIs.</li></ul> | <ul><li>Define your data model using Pydantic.</li><li>Let Instructor handle the rest.</li><li>Validated and typed data.</li></ul> |

## Installation

Get started in seconds:

```bash
pip install instructor
```

Or, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Works with all Major Providers

Instructor provides a unified API to work with any LLM provider:

```python
import instructor
from pydantic import BaseModel


class User(BaseModel):
    name: str
    age: int


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

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)
# User(name='John', age=25)
```

## Advanced Features

### Automatic Retries

Instructor automatically retries extraction attempts when validation fails, using the error message to improve future extractions.

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
    messages=[{"role": "user", "content": "John is -25 years old"}],
    max_retries=3,
)
```

### Streaming Support

Stream partial objects as they're generated, enabling real-time user experiences.

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

Extract deeply nested data structures with ease.

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

## Production-Ready

Trusted by a growing community:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

### Basic Extraction

Extract structured data with a few lines of code:

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

### Multiple Language Support

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

## Why Choose Instructor?

*   **Compared to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support without manual schema writing.
*   **Compared to LangChain/LlamaIndex:** Instructor focuses on structured extraction. It's lighter, faster, and easier to debug.
*   **Compared to Custom Solutions:** Instructor is battle-tested and handles edge cases for you.

## Contributing

We welcome contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>