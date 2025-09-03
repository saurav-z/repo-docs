# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions with Instructor, the Python library that reliably extracts structured JSON data from any language model, saving you time and reducing errors.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Define your data model using Pydantic and let Instructor handle the rest.
*   **Automatic Validation:** Ensure data integrity with built-in validation and type safety.
*   **Provider Agnostic:** Works seamlessly with OpenAI, Anthropic, Google, Ollama, and more.
*   **Automatic Retries:** Handle LLM errors gracefully with automatic retry mechanisms.
*   **Streaming Support:** Get partial results in real-time as the LLM generates them.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Production-Ready:** Trusted by 100,000+ developers and companies in production.

## Why Choose Instructor?

Instructor simplifies the process of extracting structured data from LLMs, eliminating the need for:

*   Writing complex JSON schemas
*   Manual error handling and retries
*   Parsing unstructured responses
*   Adapting to different provider APIs

## Installation

Install Instructor with a single command:

```bash
pip install instructor
```

You can also use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Get Started Quickly

### Basic Extraction Example

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

### Supported LLM Providers

Instructor works with a wide range of LLM providers:

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

## Advanced Features

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

## Languages Available

Instructor's API is available in multiple languages to support all types of development:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

## Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Instructor vs. Alternatives

**Instructor vs Raw JSON mode:** Instructor automates validation, retries, streaming, and nested objects.
**Instructor vs LangChain/LlamaIndex:** Instructor is focused and efficient, ideal for extraction.
**Instructor vs Custom solutions:** Instructor is battle-tested with a large community.

## Contribute

Join the Instructor community!  See the [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>
```
Key improvements and explanations:

*   **Strong Hook:**  The one-sentence hook clearly states the value proposition: easy and reliable structured data extraction.
*   **Clear Headings:**  Organized content with descriptive headings like "Key Features," "Why Choose Instructor?", "Installation," and "Get Started Quickly," making the information easily scannable.
*   **Bulleted Key Features:**  Highlights the main benefits in a concise, easy-to-read format.
*   **SEO-Friendly Keywords:**  Includes relevant keywords like "structured data," "LLMs," "JSON," "extraction," "Pydantic," "validation," and LLM provider names to improve search visibility.
*   **Actionable Examples:**  Provides concise, runnable code examples to demonstrate the core functionality.
*   **Benefits over Alternatives:** A dedicated section comparing Instructor to other options.
*   **Community & Support:**  Provides links to documentation, examples, the blog, and Discord for user engagement.
*   **Concise Language:** Streamlined language to make it easy to digest information quickly.
*   **Complete:** The rewritten README comprehensively covers all the key points from the original, making it a standalone resource.
*   **Maintainability:** The structure of the README makes it easier to maintain and update as the library evolves.