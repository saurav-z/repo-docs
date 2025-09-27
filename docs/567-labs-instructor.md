# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your AI workflows and get reliable, validated JSON data from any language model with Instructor.**  [See the original repository on GitHub](https://github.com/567-labs/instructor).

Instructor is a powerful Python library built on Pydantic, designed to streamline structured data extraction from Large Language Models (LLMs).  It eliminates the complexities of manual JSON parsing, error handling, and retries, allowing you to focus on building great AI applications.

**Key Features:**

*   ✅ **Pydantic Integration:** Leverage Pydantic for type safety, validation, and IDE support.
*   ✅ **Automatic Validation & Retries:**  Handles validation errors and retries extraction automatically.
*   ✅ **Multi-Provider Support:** Works seamlessly with OpenAI, Anthropic, Google, Ollama, and more.
*   ✅ **Streaming Support:** Receive partial object updates as they are generated for a better user experience.
*   ✅ **Nested Object Extraction:** Easily extract complex, nested data structures.
*   ✅ **Production-Ready:** Used by 100,000+ developers and companies for production applications.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Why Choose Instructor?

Extracting structured data from LLMs can be difficult and time-consuming. Instructor simplifies this process, saving you time and effort.

| **Challenge**                      | **Instructor Solution**                                                                  |
| :---------------------------------- | :---------------------------------------------------------------------------------------- |
| Complex JSON schema definition     | Define a simple Pydantic model                                                            |
| Validation errors                | Automatic validation and retries                                                            |
| Manual error handling & retries  |  Built-in error handling and retry mechanisms.                                           |
| Unstructured response parsing     | Extracts structured data directly into your defined models.                                |
| Provider-specific API differences |  Consistent API across all supported LLM providers.                                      |

## Get Started in Seconds

Install Instructor using pip:

```bash
pip install instructor
```

Or with your preferred package manager (uv, poetry):

```bash
uv add instructor
poetry add instructor
```

## Works with all Major LLM Providers

Use the same code with various LLM providers:

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

Instructor automatically retries when validation fails, using the error message to guide the LLM:

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

Stream partial objects as they're generated for real-time updates:

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

Easily handle complex, nested data structures:

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

## Trusted by Developers Worldwide

Instructor is the go-to choice for reliable LLM data extraction:

*   📈 **3M+ Monthly Downloads**
*   ⭐ **10K+ GitHub Stars**
*   🧑‍🤝‍🧑 **1000+ Community Contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Quick Start Guide

### Basic Extraction

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

### Multiple Language Support

Instructor offers implementations in a variety of languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

### Resources

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Why Instructor Over Alternatives?

**vs Raw JSON Mode:**  Instructor provides automatic validation, retries, streaming, and nested object support.  No manual schema writing required.

**vs LangChain/LlamaIndex:** Instructor focuses solely on structured extraction, making it lighter, faster, and easier to debug.

**vs Custom Solutions:**  Instructor is battle-tested and handles edge cases, saving you time and effort.

## Contributing

We welcome contributions!  Explore [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>