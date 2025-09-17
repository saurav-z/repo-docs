# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for seamless integration.**  [Explore the original repository](https://github.com/567-labs/instructor) for more details.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Effortless Data Extraction:** Get structured data without complex JSON parsing or error handling.
*   **Pydantic Integration:** Leverage Pydantic for type safety, validation, and IDE support.
*   **Automatic Retries:** Built-in retry mechanisms handle failed extractions, improving reliability.
*   **Streaming Support:** Receive partial objects as they're generated, enhancing responsiveness.
*   **Nested Object Handling:** Easily extract complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with a wide range of LLM providers, including OpenAI, Anthropic, Google, and local models.

## The Problem Instructor Solves

Extracting structured data from LLMs is notoriously complex, often requiring manual JSON schema creation, error handling, and retries.  Instructor simplifies this process by providing a single, intuitive interface.

**Before Instructor:**  Requires manual parsing, validation, and error handling with LLM responses.

**With Instructor:**

```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-4o-mini")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)
print(user)  # User(name='John', age=25)
```

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

## Provider Compatibility

Instructor seamlessly integrates with major LLM providers:

*   **OpenAI:** `instructor.from_provider("openai/gpt-4o")`
*   **Anthropic:** `instructor.from_provider("anthropic/claude-3-5-sonnet")`
*   **Google:** `instructor.from_provider("google/gemini-pro")`
*   **Ollama (Local):** `instructor.from_provider("ollama/llama3.2")`

You can also use API keys directly:

```python
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
```

## Production-Ready Features

### Automatic Retries

Instructor automatically retries extraction when validation fails:

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

Stream partial objects as they're generated:

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

Extract complex, nested data structures:

```python
from typing import List
from pydantic import BaseModel

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

## Trusted by Thousands

Instructor is a widely adopted library, trusted by over 100,000 developers and companies:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Getting Started

### Basic Extraction Example

Extract structured data easily:

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

Instructor is available in multiple languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

### Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Instructor vs. Alternatives

*   **vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support.
*   **vs LangChain/LlamaIndex**: Instructor is laser-focused on structured extraction for improved speed and simplicity.
*   **vs Custom Solutions**: Benefit from a battle-tested library used by thousands of developers.

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>