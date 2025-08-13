# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, structured JSON outputs with Instructor, built on Pydantic for validation, type safety, and streamlined development.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your desired data structure using Pydantic models and get structured data effortlessly.
*   **Automatic Validation:** Ensure data integrity with built-in validation and error handling.
*   **Provider Agnostic:** Works seamlessly with OpenAI, Anthropic, Google, Ollama, and more.
*   **Automatic Retries:** Handles failed extractions with intelligent retries.
*   **Streaming Support:** Receive partial objects as they're generated for real-time updates.
*   **Nested Object Support:** Easily extract complex, nested data structures.

## The Problem Instructor Solves

Extracting structured data from LLMs typically involves:

*   Writing complex JSON schemas.
*   Manually handling validation errors and retries.
*   Parsing unstructured responses.
*   Dealing with API differences across providers.

**Instructor eliminates these complexities with a simple, unified interface.**

## Get Started Quickly

### Install

```bash
pip install instructor
```

Or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Usage

Define your data model and extract structured information:

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

## Works with Major LLM Providers

Use the same code across different providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")
```

You can also directly include API keys:

```python
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
```

## Advanced Capabilities

### Automatic Retries

Instructor automatically retries failed validations:

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

## Trusted by Thousands

Instructor is trusted by:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by developers and teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Why Choose Instructor?

*   **Compared to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support without manual schema writing.
*   **Compared to LangChain/LlamaIndex:** Instructor is focused solely on structured extraction, making it lighter, faster, and easier to debug.
*   **Compared to Custom Solutions:** Instructor is battle-tested by thousands of developers and handles edge cases effectively.

## Contributing

Contributions are welcome!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>