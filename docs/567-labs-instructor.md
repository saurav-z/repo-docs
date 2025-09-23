# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON output with Instructor.**  [Explore the original repository on GitHub](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Seamless JSON Extraction:** Obtain structured data directly from any LLM without manual parsing or error handling.
*   **Pydantic Integration:** Leverage Pydantic for robust data validation, type safety, and IDE support.
*   **Automatic Retries:** Handle LLM output inconsistencies with automatic retries based on validation errors.
*   **Streaming Support:** Receive partial object updates in real-time.
*   **Nested Object Support:** Effortlessly extract complex, nested data structures.
*   **Provider Agnostic:** Works with OpenAI, Anthropic, Google, Ollama, and more, using the same API.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Instructor? The Challenge of LLM Data Extraction

Extracting structured data from LLMs is often a cumbersome process, requiring:

*   Complex JSON schema creation
*   Error handling for validation failures
*   Retries for unreliable responses
*   Manual parsing of unstructured text
*   Adaptations for diverse LLM API providers

**Instructor streamlines this process with a simple and powerful interface, saving you time and effort.**

## Instructor vs. Traditional Methods

| **Traditional Approach (Without Instructor)** | **Instructor Approach**                  |
| :------------------------------------------ | :-------------------------------------- |
| Requires manual parsing and validation.    | Simple API for immediate data retrieval. |
| Error-prone and time-consuming.            | Highly reliable and efficient.        |
| Complex code to manage edge cases.           | Handles edge cases automatically.       |

### Example: Simplify Extraction

```python
# Define your desired structure using pydantic.BaseModel
from pydantic import BaseModel
import instructor

# Instantiate an Instructor client
client = instructor.from_provider("openai/gpt-4o-mini")

# Define your data structure
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Extract data from the LLM with a simple call
product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

## Installation

Get started in seconds:

```bash
pip install instructor
```

## Production-Ready Features in Detail

### Automatic Retries

Instructor automatically retries failed validations with the error message:

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

## Used in Production By

Trusted by 100,000+ developers and companies.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Get Started

### Basic Extraction

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

### Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Why Instructor? The Superior Solution

*   **Compared to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support.
*   **Compared to LangChain/LlamaIndex:** Instructor offers a focused approach to structured extraction, leading to faster, lighter, and easier-to-debug code.
*   **Compared to Custom Solutions:** Instructor is battle-tested, handling edge cases you might not have considered.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>