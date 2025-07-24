# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, a powerful Python library built on Pydantic.** ([View the project on GitHub](https://github.com/567-labs/instructor))

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your desired output structure with Pydantic models and receive structured data effortlessly.
*   **Automated Validation:** Built-in validation ensures data integrity and type safety.
*   **Intelligent Retries:** Automatically retries failed extractions, handling common LLM issues.
*   **Streaming Support:** Receive partial objects as they are generated for improved responsiveness.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and more.
*   **Production-Ready:** Trusted by thousands of developers and companies for building robust AI applications.

## Why Choose Instructor?

Instructor streamlines the process of extracting structured data from Large Language Models, providing significant advantages over manual JSON parsing and other alternatives:

| **Challenge**                          | **Instructor Solution**                                   |
| :-------------------------------------- | :-------------------------------------------------------- |
| Complex JSON schema creation            | Define models with Pydantic, get auto-validation          |
| Validation errors                     | Automatic retries with error message integration           |
| Manual parsing and error handling       | Simple interface, less code                                |
| Inconsistent provider APIs            | Provider agnostic                                     |

## Get Started in Seconds

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

##  Seamless Integration with Multiple LLM Providers

Instructor provides a unified API, enabling you to use the same code across various LLM providers:

```python
from pydantic import BaseModel
import instructor

class User(BaseModel):
    name: str
    age: int

# OpenAI
client = instructor.from_provider("openai/gpt-4o-mini")

# Anthropic
# client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
# client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
# client = instructor.from_provider("ollama/llama3.2")

# With API keys
# client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
# client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
# client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user) # User(name='John', age=25)
```

## Advanced Features for Production

### Automatic Retries

Instructor automatically retries extraction attempts when validation errors occur, incorporating error messages for enhanced reliability:

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

# Instructor retries when validation fails
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
    max_retries=3,
)
```

### Streaming Support

Receive partial objects as they are generated, improving responsiveness:

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

Effortlessly extract complex, nested data structures:

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

## Trusted by the Community

Instructor is used by over 100,000 developers and leading companies to build AI applications. We have:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Join a community of developers at OpenAI, Google, Microsoft, AWS, and many YC startups using Instructor!

## Quick Start Guide

### Basic Data Extraction

Extract structured data from any text with ease:

```python
from pydantic import BaseModel
import instructor

class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

client = instructor.from_provider("openai/gpt-4o-mini")

product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

### Cross-Platform Availability

Instructor's user-friendly API is available in multiple languages:

-   [Python](https://python.useinstructor.com) - The original implementation
-   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
-   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
-   [Go](https://go.useinstructor.com) - Go implementation
-   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
-   [Rust](https://rust.useinstructor.com) - Rust implementation

## Explore Further

*   [Documentation](https://python.useinstructor.com) - Extensive guides and tutorials.
*   [Examples](https://python.useinstructor.com/examples/) - Ready-to-use code examples.
*   [Blog](https://python.useinstructor.com/blog/) - Learn the best practices and latest features.
*   [Discord](https://discord.gg/bD9YE9JArw) - Connect with the community and get support.

## Instructor vs. Alternatives

*   **Raw JSON Mode**: Instructor adds automatic validation, retries, streaming, and handles nested objects.
*   **LangChain/LlamaIndex**: Instructor excels in structured extraction, offering a lightweight, efficient solution.
*   **Custom Solutions**: Leverage Instructor's battle-tested, production-ready features and handle edge cases effortlessly.

## Contributing

We welcome contributions!  Review our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to contribute.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>