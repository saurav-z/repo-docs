# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, structured JSON output with Instructor, built on Pydantic for type safety, validation, and seamless integration.**  Learn more on the original repo: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Eliminate the need for complex JSON schemas and manual parsing.
*   **Pydantic Integration:** Leverage Pydantic for robust validation, type safety, and IDE support.
*   **Automatic Retries:** Automatically retry failed extractions, improving reliability.
*   **Streaming Support:** Stream partial objects as they're generated for faster results.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, and more).
*   **Production-Ready:** Built-in features to handle common LLM extraction challenges.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## The Challenge of Structured Data Extraction

Getting structured data from LLMs can be a complex process, often requiring you to:

*   Create and maintain complex JSON schemas.
*   Handle validation errors manually.
*   Implement retry mechanisms for failed extractions.
*   Parse unstructured responses effectively.
*   Adapt to different provider APIs.

**Instructor simplifies all of this with a single, intuitive interface.**

| **Without Instructor**                                                                                                                            | **With Instructor**                                                                                                                                                                                                         |
| :------------------------------------------------------------------------------------------------------------------------------------------------ | :---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python  response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[ { "type": "function", "function": { "name": "extract_user", "parameters": { "type": "object", "properties": { "name": {"type": "string"}, "age": {"type": "integer"}, }, }, }, } ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass  ``` | ```python  client = instructor.from_provider("openai/gpt-4")  user = client.chat.completions.create(  response_model=User,  messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Installation

Get started in seconds with your preferred package manager:

```bash
pip install instructor
```

Or with your package manager:
```bash
uv add instructor
poetry add instructor
```

## Supported LLM Providers

Instructor offers seamless integration with a wide range of LLM providers, using the same API:

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

## Powerful Production-Ready Features

### Automatic Retries

Instructor automatically retries extractions when validation fails, improving the reliability of your applications:

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

Stream partial objects as they're generated for a more responsive user experience:

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

Effortlessly extract complex, nested data structures:

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

## Proven in Production

Instructor is trusted by over 100,000 developers and companies building AI applications, including teams at OpenAI, Google, Microsoft, AWS, and many YC startups:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Get Started

### Basic Extraction

Quickly extract structured data from any text:

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

Instructor's straightforward API is accessible in multiple programming languages:

*   [Python](https://python.useinstructor.com) - The original implementation.
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides.
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes.
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices.
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community.

## Why Choose Instructor?

*   **vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support, eliminating manual schema creation.
*   **vs LangChain/LlamaIndex**: Instructor is laser-focused on structured extraction, offering a lighter, faster, and more debuggable solution.
*   **vs Custom solutions**: Benefit from a battle-tested solution, proven by thousands of developers, handling edge cases you might not have considered.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>