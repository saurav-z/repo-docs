# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your interactions with large language models by reliably extracting structured data, eliminating the need for complex JSON parsing and error handling.** Get started at the [Instructor GitHub Repository](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Define Python models (using Pydantic) and effortlessly extract structured data from LLM responses.
*   **Provider Agnostic:** Works seamlessly with major LLM providers, including OpenAI, Anthropic, Google, and local models (Ollama).
*   **Automatic Error Handling:** Includes automatic retry mechanisms for failed extractions and validation errors.
*   **Streaming Support:** Stream partial objects for real-time processing as they're generated.
*   **Nested Object Support:** Easily extract and handle complex, nested data structures.
*   **Production-Ready:** Battle-tested and trusted by over 100,000 developers and companies.

## Getting Started

### Installation

Install Instructor using pip:

```bash
pip install instructor
```

Or use your preferred package manager (uv, poetry, etc.):

```bash
uv add instructor
poetry add instructor
```

### Basic Usage Example

Define your data model and extract data with a few lines of code:

```python
from pydantic import BaseModel
import instructor

# Define your data model
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Initialize the Instructor client
client = instructor.from_provider("openai/gpt-4o-mini")

# Extract data from a text prompt
product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Output: Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

## Why Choose Instructor?

Instructor provides a streamlined approach to structured data extraction compared to alternatives:

*   **Effortless Data Extraction:** No manual schema writing.
*   **Handles Common Challenges:** Automatic validation, retries, streaming, and nested object support.
*   **Focused and Efficient:** Lightweight and easy to debug, in contrast to more complex frameworks.
*   **Battle-Tested:** Trusted by a large community for robust and reliable extraction.

## Production-Ready Features in Detail

### Automatic Retries

Handle validation failures automatically:

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
    max_retries=3,  # Instructor automatically retries when validation fails
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

## Used in Production By

Instructor is trusted by a large and growing community:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Provider Support

Instructor supports all major LLM providers, offering a consistent API:

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

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>