# Instructor: Effortlessly Get Structured Data from LLMs

**Simplify LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for type safety and ease of use.**  ([View the original repo](https://github.com/567-labs/instructor))

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Effortless Data Extraction:** Define your desired output model, and Instructor handles the rest.
*   **Pydantic Integration:** Leverage Pydantic for validation, type safety, and IDE support.
*   **Automatic Retries:** Built-in retry mechanism for failed extractions, ensuring reliable results.
*   **Streaming Support:** Receive and process partial data as it's generated for improved responsiveness.
*   **Nested Object Handling:** Easily extract complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and Ollama.

## The Problem Instructor Solves

Extracting structured data from Large Language Models (LLMs) can be complex and time-consuming. Instructor simplifies this process by eliminating the need for:

*   Writing and managing complex JSON schemas.
*   Manually handling validation errors.
*   Implementing retry mechanisms for failed extractions.
*   Parsing unstructured responses.
*   Dealing with the inconsistencies of different provider APIs.

## Getting Started

### Installation

Install Instructor with pip:

```bash
pip install instructor
```

or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Usage Example

Define a Pydantic model and extract data:

```python
from pydantic import BaseModel
import instructor

# Define your model
class Product(BaseModel):
    name: str
    price: float
    in_stock: bool

# Initialize the client
client = instructor.from_provider("openai/gpt-4o-mini")

# Extract structured data
product = client.chat.completions.create(
    response_model=Product,
    messages=[{"role": "user", "content": "iPhone 15 Pro, $999, available now"}],
)

print(product)
# Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

## Production-Ready Features

### Automatic Retries

Instructor automatically retries when validation fails, incorporating the error message for improved accuracy.

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

## Works with All Major Providers

Use the same code across providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")
```

## Used in Production by

Trusted by over 100,000 developers and companies, with over 3 million monthly downloads.

## Why Choose Instructor?

Instructor offers significant advantages over alternatives:

*   **JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support.
*   **LangChain/LlamaIndex:** Instructor's focus on structured extraction makes it lighter, faster, and easier to debug.
*   **Custom Solutions:** Instructor is battle-tested by thousands of developers and handles edge cases.

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

We welcome contributions!  See our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get involved.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>