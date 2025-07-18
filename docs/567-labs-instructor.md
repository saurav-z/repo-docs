# Instructor: Effortlessly Extract Structured Data from LLMs

**Instructor empowers developers to reliably extract structured JSON data from any Large Language Model (LLM), eliminating the complexities of parsing, validation, and retries.** Get started with Instructor on [GitHub](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define Python models using Pydantic, and Instructor handles the extraction.
*   **Automatic Validation:** Ensures data integrity with built-in validation based on your Pydantic models.
*   **Intelligent Retries:** Automatically retries failed extractions, increasing reliability.
*   **Streaming Support:** Receive partial objects in real-time as the LLM generates data.
*   **Nested Object Support:** Easily handle complex, nested data structures.
*   **Cross-Provider Compatibility:** Works seamlessly with leading LLM providers like OpenAI, Anthropic, Google, and local models via Ollama.
*   **Multiple Language Support:** Implementations available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## The Challenge: Structured Data Extraction

Extracting structured data from LLMs can be a daunting task, requiring you to:

1.  Craft complex JSON schemas.
2.  Manage validation errors.
3.  Implement retry mechanisms.
4.  Parse unstructured responses.
5.  Adapt to various provider APIs.

**Instructor streamlines this process with a simple interface:**

| **Without Instructor**                                                                                                                                   | **With Instructor**                                                                                                                          |
| ------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python                                                                                                                                                 response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[{"type": "function", "function": {"name": "extract_user", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"},},},},}] ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

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

## Provider Compatibility

Use the same code across a variety of LLM providers:

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

## Production-Ready Features

### Automatic Retries

Instructor automatically retries failed validation attempts:

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

Easily extract complex, nested data:

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

## Used by Thousands

Trusted by 100,000+ developers and companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

### Basic Extraction Example

Extract structured data from natural language:

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

Instructor is available in multiple languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

## Why Instructor?

**Compared to Raw JSON Mode**: Instructor offers automatic validation, retries, streaming, and nested object support, eliminating the need for manual schema writing.

**Compared to LangChain/LlamaIndex**: Instructor is laser-focused on structured extraction, offering a lighter, faster, and more debuggable solution.

**Compared to Custom Solutions**: Instructor provides a battle-tested solution, used by thousands of developers and able to handle complex edge cases.

## Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Contributing

We welcome contributions! Review our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>