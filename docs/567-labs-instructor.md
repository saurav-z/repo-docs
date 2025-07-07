# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable JSON outputs with Instructor, built on Pydantic for type safety, validation, and ease of use.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features:

*   **Simplified Data Extraction:** Eliminate the need for complex JSON schemas and manual parsing.
*   **Pydantic Integration:** Leverage the power of Pydantic for data validation, type safety, and seamless IDE support.
*   **Automatic Retries:** Handles LLM response errors with built-in retry mechanisms.
*   **Streaming Support:** Receive partial data as it's generated, improving responsiveness.
*   **Nested Object Support:** Easily extract complex, nested data structures from LLM responses.
*   **Multi-Provider Compatibility:** Works with leading LLM providers like OpenAI, Anthropic, Google, and Ollama.

## The Challenge: Extracting Structured Data from LLMs

Getting structured data from LLMs can be a cumbersome process, often involving:

1.  Writing and maintaining complex JSON schemas.
2.  Handling validation errors.
3.  Implementing retry mechanisms for failed extractions.
4.  Manually parsing unstructured responses.
5.  Adapting to different LLM provider APIs.

**Instructor streamlines this process with a single, intuitive interface:**

| **Before Instructor**                                  | **After Instructor**                                  |
| ------------------------------------------------------- | ------------------------------------------------------- |
| Requires manual JSON parsing, schema definition, and error handling |  Uses Pydantic models for type safety and validation |
|  Tedious and error-prone. | Effortless data extraction. |

## Installation: Get Started Quickly

```bash
pip install instructor
```

Alternatively, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Seamless Integration with Major LLM Providers

Use the same Instructor code with any LLM provider:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# All use the same API!
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-Ready Features for Robust Applications

### Automatic Retries

Instructor automatically retries failed validations, using the error message to improve future extractions.

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

Stream partial objects in real-time as they are generated.

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

Extract complex nested data structures effortlessly.

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

## Trusted by a Growing Community

Instructor is the choice of over 100,000 developers and companies for building AI applications.

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started Today

### Basic Extraction

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

### Multi-Language Support

Instructor is available in many languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive Guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-Paste Recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and Best Practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Choose Instructor?

*   **Over Raw JSON Mode:** Instructor offers automatic validation, retries, streaming, and nested object support without the need for manual schema writing.
*   **Over LangChain/LlamaIndex:** Instructor focuses on a single purpose - structured extraction, making it lighter, faster, and easier to debug.
*   **Over Custom Solutions:** Benefit from a battle-tested solution used by thousands of developers, designed to handle complex edge cases.

## Contributing

We welcome your contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>