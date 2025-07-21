# Instructor: Effortlessly Get Structured JSON from LLMs

**Simplify your interactions with Large Language Models (LLMs) and reliably extract structured data with Instructor, built on Pydantic for type safety and ease of use.**

[Visit the original repository](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   ✅ **Seamless Data Extraction:** Converts natural language text from any LLM into structured JSON, eliminating the need for manual parsing and error handling.
*   ✅ **Pydantic Integration:** Leverages Pydantic for type validation, ensuring data integrity and providing robust IDE support.
*   ✅ **Automatic Error Handling & Retries:** Automatically handles validation failures and retries extractions, making your application more reliable.
*   ✅ **Flexible Provider Support:** Works with major LLM providers like OpenAI, Anthropic, Google, and local models (Ollama) using a consistent API.
*   ✅ **Streaming Support:** Allows you to stream partial objects as they are generated, improving user experience and efficiency.
*   ✅ **Nested Object Extraction:** Effortlessly handles complex, nested data structures.
*   ✅ **Multi-language support:** Use Instructor's simple API in multiple languages, including Python, TypeScript, Ruby, Go, Elixir, and Rust.

## The Challenge of Structured Data Extraction

Getting structured data from LLMs is often a complex process that typically involves:

1.  Creating intricate JSON schemas.
2.  Dealing with validation errors.
3.  Implementing retry mechanisms for failed extractions.
4.  Parsing unstructured responses.
5.  Managing various API differences across LLM providers.

**Instructor simplifies this with an intuitive interface:**

| **Before Instructor**                                                                                                                              | **After Instructor**                                                                                                                            |
| :------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```python response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[{"type": "function", ...}], ) # Parse, Validate, Handle Errors... ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], ) # That's it! ``` |

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or with other package managers:

```bash
uv add instructor
poetry add instructor
```

## Provider Support

Instructor offers a unified API across all major LLM providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# With API keys directly
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-Ready Features

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

## Trusted by the AI Community

Join the community of over 100,000 developers and companies who rely on Instructor to build AI applications:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Companies using Instructor include OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Get Started

### Basic Extraction

Extract structured data from any text:

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

## Why Instructor?

*   **Superior to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating manual schema creation.
*   **Lighter and Faster than Alternatives (like LangChain/LlamaIndex):** Instructor's focused design makes it faster and easier to debug.
*   **Battle-Tested and Reliable:** Instructor handles complex edge cases, saving you time and effort.

## Contributing

Contributions are welcome! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to start.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>