# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your interactions with Large Language Models and get reliable, validated JSON output with Instructor, built on Pydantic.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Define your desired data structure with Pydantic models and let Instructor handle the extraction.
*   **Automatic Validation:** Ensures data integrity with built-in Pydantic validation, catching errors early.
*   **Intelligent Retries:** Automatically retries failed extractions, making your application more robust.
*   **Streaming Support:** Get partial objects in real-time as the LLM generates output, for a more responsive user experience.
*   **Nested Object Support:**  Easily extract complex, nested data structures with a clean and intuitive API.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and local models via Ollama.
*   **Multi-Language Support:** Available in Python (original), TypeScript, Ruby, Go, Elixir, and Rust.

## The Problem Instructor Solves

Getting structured data from LLMs can be a complex process, often involving:

*   Writing intricate JSON schemas.
*   Handling validation errors manually.
*   Implementing retry logic for unreliable extractions.
*   Parsing unstructured responses.
*   Adapting to different LLM provider APIs.

**Instructor streamlines the entire process with a simple and efficient interface.**

### Before Instructor vs. After Instructor

| **Without Instructor**                                                                                                                                                                                     | **With Instructor**                                                                                                                                                                  |
| ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python                                                                                                                                                                                                 | ```python                                                                                                                                                                             |
| response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[ { "type": "function", "function": { "name": "extract_user", "parameters": { "type": "object", "properties": { "name": {"type": "string"}, "age": {"type": "integer"}, }, }, }, } ],) | client = instructor.from_provider("openai/gpt-4")                                                                                                                                 |
| # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass                | user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], )                                                                            |
|                                                                                                                                                                                                            | # That's it! user is validated and typed                                                                                                                                                |

## Installation

Get started in seconds:

```bash
pip install instructor
```

Alternatively, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Supported LLM Providers

Instructor provides a consistent API across major LLM providers, making it easy to switch or support multiple providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# All use the same API!
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-Ready Features

### Automatic Retries

Instructor automatically retries failed validations, using error messages to guide the LLM:

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

Stream partial objects as they're generated, enabling real-time data updates:

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

Extract complex, nested data with ease:

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

Instructor is a battle-tested tool trusted by over 100,000 developers and companies:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Getting Started

### Basic Extraction

Extract structured data from text in a few lines of code:

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

###  Multi-Language Support

Instructor's simple API is available in multiple languages for cross-platform support:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

## Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Choose Instructor?

*   **Superior to Raw JSON Mode:** Provides automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.
*   **More Focused than Alternatives (LangChain/LlamaIndex):** Instructor is specialized for structured extraction, offering a lighter, faster, and easier-to-debug solution.
*   **Production-Ready and Reliable:** Battle-tested by thousands of developers, handling edge cases and providing a robust solution.

## Contributing

Contributions are welcome!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>