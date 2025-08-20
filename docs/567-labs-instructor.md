# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor – no more manual parsing or error handling.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Define your data model using Pydantic and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity with built-in validation based on your Pydantic models.
*   **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   **Streaming Support:** Get partial results as they are generated for a more responsive experience.
*   **Nested Object Support:** Handle complex, nested data structures with ease.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and Ollama.
*   **Production-Ready:** Used by thousands of developers and companies in production.

## The Problem Instructor Solves

Working with LLMs to extract structured data can be complex and error-prone. Instructor streamlines this process by:

*   Eliminating the need for writing complex JSON schemas.
*   Automating validation error handling.
*   Simplifying the handling of different provider APIs.

## Get Started in Seconds

Install Instructor using pip:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Example: Basic Extraction

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

## Key Advantages Over Alternatives

**Instructor offers significant advantages over alternative approaches:**

*   **Compared to Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating the need for manual schema writing.
*   **Compared to LangChain/LlamaIndex:** Instructor focuses solely on structured extraction, making it lighter, faster, and easier to debug.
*   **Compared to Custom solutions:** Instructor is battle-tested by thousands of developers and handles edge cases you might not have considered.

## Production-Ready Features in Detail

### Automatic Retries

Automatically retry extractions when validation fails, improving the reliability of your application.

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

Stream partial objects as they're generated to improve responsiveness.

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

Extract complex nested data structures with ease.

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

## Supported LLM Providers

Instructor seamlessly works with popular LLM providers:

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

## Used in Production By

Instructor is trusted by over 100,000 developers and companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Additional Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>