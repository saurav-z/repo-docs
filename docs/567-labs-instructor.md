# Instructor: Simplify LLM Output with Structured Data Extraction

**Effortlessly extract structured data from any LLM, making integration and validation seamless with Python and other languages.** Learn more about Instructor on its [original GitHub repository](https://github.com/567-labs/instructor).

Instructor streamlines the process of obtaining structured data from large language models (LLMs), eliminating the need for complex JSON parsing, error handling, and manual retries. Built on Pydantic, Instructor provides type safety and IDE support, enabling developers to focus on building applications rather than wrestling with LLM outputs.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simple Interface:** Define your desired data structure with Pydantic models.
*   **Automatic Validation:** Ensures data integrity with built-in validation using Pydantic.
*   **Automated Retries:** Automatically retries failed extractions due to validation errors.
*   **Streaming Support:** Receive partial objects as they are generated.
*   **Nested Object Support:** Handles complex, nested data structures effortlessly.
*   **Multi-Provider Support:** Works seamlessly with popular LLM providers like OpenAI, Anthropic, Google, and local LLMs (Ollama).
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Use Instructor?

Instructor simplifies LLM integration by addressing the common challenges of structured data extraction:

*   **Simplifies JSON schema creation:** No need to write complex JSON schemas.
*   **Automates error handling and retries:** Handles validation errors and retries failed extractions.
*   **Provides type safety and IDE support:** Leveraging Pydantic for robust data validation and developer productivity.
*   **Supports various LLM providers:** Works with popular LLM providers and local LLMs.

**Instructor offers a more streamlined and efficient approach compared to manual parsing, custom solutions, or frameworks focused on other tasks.**

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

## Getting Started

### Basic Extraction

Extract structured data effortlessly:

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

### Automatic Retries

Handle validation errors with ease:

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

Receive partial objects as they are generated:

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

Easily extract complex nested data:

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

## Production-Ready Features

Instructor offers powerful features for production environments:

*   **Automatic retries** to handle failed validations.
*   **Streaming support** to receive data incrementally.
*   **Nested object support** for complex data structures.

## Used in Production By

Instructor is trusted by over 100,000 developers and companies, including teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

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