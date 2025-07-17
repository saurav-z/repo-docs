# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable JSON outputs with Instructor, a powerful library built on Pydantic.**

[Get Started with Instructor](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Define your desired data structure using Pydantic models, and Instructor handles the rest.
*   **Automatic Validation & Error Handling:** Eliminate manual JSON parsing and error handling with built-in validation and retry mechanisms.
*   **Provider Agnostic:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and local models (Ollama).
*   **Automatic Retries:** Automatically retry failed extractions based on validation errors, improving reliability.
*   **Streaming Support:** Receive partial objects in real-time as the LLM generates output.
*   **Nested Object Support:** Effortlessly handle complex, nested data structures.

## The Problem: Data Extraction Challenges

Extracting structured data from LLMs traditionally requires:

*   Writing complex JSON schemas
*   Handling validation errors
*   Implementing retry logic
*   Parsing unstructured responses
*   Adapting to different provider APIs

## The Solution: Instructor's Simple Interface

Instructor simplifies the process with a clean, intuitive interface:

| Without Instructor                                                    | With Instructor                                                    |
| :-------------------------------------------------------------------- | :----------------------------------------------------------------- |
| ```python                                                            | ```python                                                           |
| response = openai.chat.completions.create(                           | client = instructor.from_provider("openai/gpt-4")                 |
|     model="gpt-4",                                                   |                                                                    |
|     messages=[{"role": "user", "content": "..."}],                   | user = client.chat.completions.create(                                |
|     tools=[                                                            |     response_model=User,                                            |
|         {                                                              |     messages=[{"role": "user", "content": "..."}],                  |
|             "type": "function",                                        | )                                                                    |
|             "function": {                                              |                                                                    |
|                 "name": "extract_user",                                | # That's it! user is validated and typed                           |
|                 "parameters": {                                        | ```                                                                 |
|                     "type": "object",                                  |                                                                    |
|                     "properties": {                                    |                                                                    |
|                         "name": {"type": "string"},                     |                                                                    |
|                         "age": {"type": "integer"},                    |                                                                    |
|                     },                                                 |                                                                    |
|                 },                                                     |                                                                    |
|             },                                                         |                                                                    |
|         }                                                              |                                                                    |
|     ],                                                                |                                                                    |
| )                                                                     |                                                                    |
|                                                                        |                                                                    |
| # Parse response                                                      |                                                                    |
| tool_call = response.choices[0].message.tool_calls[0]               |                                                                    |
| user_data = json.loads(tool_call.function.arguments)                  |                                                                    |
|                                                                        |                                                                    |
| # Validate manually                                                   |                                                                    |
| if "name" not in user_data:                                           |                                                                    |
|     # Handle error...                                                  |                                                                    |
|     pass                                                               |                                                                    |
| ```                                                                    |                                                                    |

## Installation

Install Instructor in seconds:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Supported LLM Providers

Use the same code across various providers:

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

## Advanced Features

### Automatic Retries

Instructor automatically retries failed extractions:

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

## Join the Instructor Community

Instructor is trusted by over 100,000 developers and leading companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Get Started Quickly

### Basic Extraction Example

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

## Why Choose Instructor?

**Instructor vs. Raw JSON Mode:** Instructor offers automatic validation, retries, streaming, and nested object support. No manual schema writing is required.

**Instructor vs. LangChain/LlamaIndex:** Instructor specializes in structured extraction. It's lighter, faster, and easier to debug.

**Instructor vs. Custom Solutions:** Benefit from a battle-tested solution used by thousands of developers. Instructor handles edge cases.

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>