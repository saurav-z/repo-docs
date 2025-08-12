# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify LLM interactions and reliably extract structured data like JSON with Instructor, a Python library built on Pydantic.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified API:** Define your desired data structure with Pydantic and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity and type safety with built-in validation using Pydantic.
*   **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   **Streaming Support:**  Stream partial objects as they're generated, enabling real-time applications.
*   **Nested Object Support:** Effortlessly extract complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with leading LLM providers including OpenAI, Anthropic, Google, and local models like Ollama.

## Why Choose Instructor?

Instructor streamlines the process of extracting structured data from LLMs, eliminating the need for complex schema definitions, manual parsing, error handling, and retries.  It handles these complexities, providing a clean and efficient interface for developers.

```python
# Example of Instructor in action
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

**Compared to alternative approaches, Instructor offers:**

*   **Simplicity:** Eliminates the need for manual JSON parsing and error handling.
*   **Efficiency:** Streamlines data extraction, saving development time.
*   **Reliability:** Automatic retries and validation ensure data integrity.
*   **Flexibility:** Supports a wide range of LLM providers and data structures.

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## LLM Provider Compatibility

Instructor seamlessly integrates with all major LLM providers:

```python
client = instructor.from_provider("openai/gpt-4o") # OpenAI
client = instructor.from_provider("anthropic/claude-3-5-sonnet") # Anthropic
client = instructor.from_provider("google/gemini-pro") # Google
client = instructor.from_provider("ollama/llama3.2") # Ollama (local)

# With API keys directly (no environment variables needed)
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

# All use the same API!
```

## Production-Ready Features

### Automatic Retries

Instructor automatically retries failed extractions when validation fails, using error messages for improved reliability:

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

Extract complex, nested data structures with ease:

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

## Trusted by the Community

Instructor is used by over 100,000 developers and companies, and the community is thriving:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies leveraging Instructor include teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Get Started

### Basic Extraction

Extract structured data from unstructured text in a flash:

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

Instructor's simple API is available in multiple programming languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to start contributing.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>