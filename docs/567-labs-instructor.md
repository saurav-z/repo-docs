# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your interactions with Large Language Models and get reliable JSON outputs every time with Instructor.**

[Visit the original repo](https://github.com/567-labs/instructor)

Instructor is a powerful Python library built on Pydantic that streamlines the process of extracting structured data from any LLM, providing validation, type safety, and robust error handling.

## Key Features

*   **Simplified Extraction:** Define your desired data model using Pydantic and let Instructor handle the rest.
*   **Automatic Validation & Retries:** Ensures data integrity with automatic validation and retries for failed extractions.
*   **Broad Provider Compatibility:** Works seamlessly with leading LLM providers, including OpenAI, Anthropic, Google, and Ollama.
*   **Streaming Support:** Stream partial objects as they're generated for a more responsive user experience.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Get Started

### Installation

Install Instructor with pip:

```bash
pip install instructor
```

Or with your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Usage

Effortlessly extract structured data from any text:

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

## Why Choose Instructor?

Instructor offers a superior experience compared to other solutions:

*   **Compared to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating the need for manual schema writing and error handling.
*   **Compared to LangChain/LlamaIndex:** Instructor focuses solely on structured extraction, making it lightweight, faster, and easier to debug.
*   **Compared to Custom Solutions:** Benefit from a battle-tested library used by thousands of developers, designed to handle a wide range of edge cases.

## Production-Ready Features

### Automatic Retries

Instructor automatically retries extractions when validation fails:

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

## Used in Production By

Trusted by over 100,000 developers and companies building AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Resources

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