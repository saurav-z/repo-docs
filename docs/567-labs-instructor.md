# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, the Python library built for structured data extraction.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your desired data structure with Pydantic models and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity with built-in Pydantic validation and type safety.
*   **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   **Streaming Support:** Receive partial objects in real-time as the LLM generates them.
*   **Nested Object Support:** Easily handle complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and Ollama.

## The Problem: Why Instructor?

Extracting structured data from Large Language Models (LLMs) is often a complex and error-prone process, involving:

*   Writing intricate JSON schemas
*   Handling validation errors
*   Implementing retry mechanisms
*   Parsing unstructured responses
*   Adapting to different provider APIs

**Instructor streamlines this process, offering a clean and efficient solution.**

| **Without Instructor** | **With Instructor** |
| ---------------------- | -------------------- |
| (See original README for code example) | (See original README for code example) |

## Installation

Get started in seconds:

```bash
pip install instructor
```

Alternatively, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Use Instructor with Any Major Provider

Instructor offers a unified API across all major LLM providers, including:

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

## Production-Ready Features

### Automatic Retries

Instructor automatically retries failed validation, incorporating error messages:

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

Receive partial objects as they're generated:

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

Extract complex, nested data structures:

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

## Trusted by the Community

Instructor is a proven solution, powering AI applications for:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Adopted by developers and companies including those at OpenAI, Google, Microsoft, AWS, and YC startups.

## Get Started

### Basic Extraction Example

Extract structured data quickly and easily:

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

Instructor's API is available in a growing number of languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

### Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Comparison: Instructor vs. Alternatives

*   **vs Raw JSON mode:** Instructor offers automatic validation, retries, streaming, and nested object support without manual schema definition.
*   **vs LangChain/LlamaIndex:** Instructor focuses solely on structured extraction, making it more lightweight, faster, and easier to debug.
*   **vs Custom solutions:** Instructor is a battle-tested library used by thousands of developers, handling edge cases and complexities that you might miss.

## Contributing

We welcome contributions! Check out the [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>