# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your LLM interactions by effortlessly extracting reliable, validated JSON and structured data from any language model with Instructor, built on Pydantic.**

[View the project on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define your desired data structure with Pydantic models and let Instructor handle the extraction.
*   **Automatic Validation:** Ensures data integrity with built-in validation based on your Pydantic models.
*   **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   **Streaming Support:** Stream partial data as it's generated for faster feedback.
*   **Nested Object Support:** Easily extract complex, hierarchical data structures.
*   **Multi-Provider Compatibility:** Works seamlessly with a wide range of LLM providers, including OpenAI, Anthropic, Google, and local models like Ollama.
*   **Production-Ready:** Trusted by over 100,000 developers and companies, handling real-world use cases.

## Why Instructor?

Extracting structured data from LLMs can be a complex and error-prone process, often involving:

*   Writing and managing complex JSON schemas
*   Handling validation errors manually
*   Implementing retry mechanisms for failed extractions
*   Parsing unstructured responses
*   Adapting to different provider APIs

**Instructor eliminates these complexities with a simple, intuitive interface:**

| **Without Instructor**                                                                     | **With Instructor**                                                                    |
| ------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------- |
| ```python response = openai.chat.completions.create(model="gpt-4",messages=[...],tools=[...]) user_data = json.loads(tool_call.function.arguments) if "name" not in user_data: # Handle error... pass ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create(response_model=User,messages=[...]) # That's it! ``` |

## Getting Started

### Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or with your preferred package manager:
```bash
uv add instructor
poetry add instructor
```

### Basic Usage

Define your data model and extract structured data:

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

## Powerful Features

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

## Provider Support

Instructor supports major LLM providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

# With API keys directly
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")
```

## Language Support

Instructor's simple API is available in multiple languages:

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

## Why Instructor?

**Instructor offers significant advantages over alternatives:**

*   **vs Raw JSON mode:** Provides automatic validation, retries, streaming, and nested object support. No manual schema writing.
*   **vs LangChain/LlamaIndex:** Focused solely on structured extraction. It's lighter, faster, and easier to debug.
*   **vs Custom solutions:** Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>