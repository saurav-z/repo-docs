# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your LLM interactions and reliably extract structured data with Instructor, a powerful tool built on Pydantic.**

[Go to the original repository](https://github.com/567-labs/instructor)

## Key Features

*   **Simplified Data Extraction:** Define your desired data structure using Pydantic models and let Instructor handle the rest.
*   **Automatic Validation:** Ensure data integrity with built-in validation, reducing errors and improving reliability.
*   **Robust Error Handling:** Instructor automatically retries failed extractions, increasing success rates and handling edge cases.
*   **Streaming Support:** Receive partial objects as they're generated, enabling real-time applications.
*   **Nested Object Support:** Easily extract complex, nested data structures without manual parsing.
*   **Provider Agnostic:** Works seamlessly with major LLM providers, using the same API.
*   **Production-Ready:** Designed for real-world applications with automatic retries, streaming, and nested object support.

## Why Instructor? The Problem It Solves

Extracting structured data from LLMs can be complex and error-prone, often requiring:

*   Writing intricate JSON schemas
*   Handling validation errors manually
*   Implementing retry mechanisms
*   Parsing unstructured responses
*   Adapting to different API providers

**Instructor simplifies this process with a single, intuitive interface.**

| **Before Instructor**                                                                                                                                                                                                                                               | **With Instructor**                                                                                                                                                                                                                         |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```python                                                                                                                                                                                                                                                          | ```python                                                                                                                                                                                                                               |
| `response = openai.chat.completions.create(  model="gpt-4",  messages=[{"role": "user", "content": "..."}],  tools=[  {"type": "function",  "function": {  "name": "extract_user",  "parameters": {  "type": "object",  "properties": {  "name": {"type": "string"},  "age": {"type": "integer"},  },  },  }  ],  )` | `client = instructor.from_provider("openai/gpt-4")  user = client.chat.completions.create(  response_model=User,  messages=[{"role": "user", "content": "..."}],  )`                                                              |
| `tool_call = response.choices[0].message.tool_calls[0]  user_data = json.loads(tool_call.function.arguments)  if "name" not in user_data:  # Handle error...  pass`                                                                                               | `# That's it! user is validated and typed`                                                                                                                                                                                                  |

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or with your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Supported LLM Providers

Instructor works with all major LLM providers:

*   OpenAI
*   Anthropic
*   Google
*   Ollama (Local)
*   Groq

You can use the same code regardless of the provider. API keys can be passed directly for convenience:

```python
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
```

## Core Capabilities and Production-Ready Features

### Automatic Retries

Instructor automatically retries extractions when validation fails, increasing success rates:

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

Stream partial objects as they are generated:

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

## Adoption & Trust

Instructor is trusted by a large and active community and deployed in production by numerous companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used in production by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Quickstart Guide

### Basic Extraction

Extract structured data from any text:

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

### Multiple Language Support

Instructor's simple API is available in many languages:

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

## Instructor vs. Alternatives

**vs Raw JSON mode**: Instructor offers automatic validation, retries, streaming, and nested object support, without the need for manual schema creation.

**vs LangChain/LlamaIndex**: Instructor is focused on structured extraction, offering a more lightweight, efficient, and easily debuggable solution.

**vs Custom solutions**: Benefit from a battle-tested library refined by thousands of developers, handling complex use cases and edge cases out of the box.

## Contributing

We welcome contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>