# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and reliably extract structured data with Instructor, a powerful library built on Pydantic.**  Access the [original repo here](https://github.com/567-labs/instructor).

Instructor makes it easy to get reliable JSON from any LLM, handling complex tasks like schema validation, error handling, and retries.

**Key Features:**

*   **Simple Interface:** Define your desired output with Pydantic models and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity with built-in Pydantic validation.
*   **Intelligent Retries:** Automatically retries failed extractions, increasing reliability.
*   **Streaming Support:**  Receive partial objects as they're generated, improving responsiveness.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and Ollama.
*   **Multi-Language Support:** Implementations in Python, TypeScript, Ruby, Go, Elixir, and Rust.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Why Instructor?

Extracting structured data from Large Language Models (LLMs) is often a difficult process. Without Instructor, you're facing a series of complex tasks:

1.  Crafting intricate JSON schemas
2.  Managing and handling validation errors
3.  Implementing retry mechanisms for failed extractions
4.  Parsing unstructured responses to extract data
5.  Adapting to the API nuances of different providers

**Instructor simplifies this process with a single, intuitive interface:**

| **Without Instructor**                                                                                                                                   | **With Instructor**                                                                                                                                   |
| :-------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[{ "type": "function", "function": { "name": "extract_user", "parameters": { "type": "object", "properties": { "name": {"type": "string"}, "age": {"type": "integer"}, }, }, }, }], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Getting Started - Installation

Install Instructor quickly with your preferred package manager:

```bash
pip install instructor
```

Alternatively:

```bash
uv add instructor
poetry add instructor
```

## Seamless Provider Integration

Use the same code with any major LLM provider:

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

Instructor automatically retries extractions when validation fails, incorporating the error message for better results:

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

Get partial objects as they're generated for improved responsiveness:

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

Extract complex and nested data structures with ease:

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

## Trusted by Thousands

Instructor is a preferred tool for many developers and companies creating advanced AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used in production by teams at OpenAI, Google, Microsoft, AWS, and many Y Combinator startups.

## Quick Start Guide

### Basic Extraction

Extract structured data from any text with minimal code:

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

Instructor's intuitive API is available in several languages:

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

**Compared to Raw JSON mode:** Instructor offers automated validation, retries, streaming, and support for nested objects, without requiring manual schema creation.

**Compared to LangChain/LlamaIndex:** Instructor focuses on structured extraction, providing a lighter, faster, and more easily debugged solution.

**Compared to Custom solutions:**  Instructor has been proven by thousands of developers, handling edge cases and complexities that you may not have anticipated.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get involved.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>