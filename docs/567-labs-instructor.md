# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, validated JSON with Instructor, built on Pydantic.**  Visit the original repository [here](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Define Pydantic models and get structured data without JSON parsing.
*   **Automatic Validation:** Built-in validation using Pydantic ensures data integrity.
*   **Automatic Retries:** Handles extraction failures and retries automatically.
*   **Streaming Support:** Stream partial objects as they're generated.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Cross-Provider Compatibility:** Works seamlessly with major LLM providers (OpenAI, Anthropic, Google, Ollama, etc.).
*   **IDE Support:** Leverage Pydantic for type safety and IDE autocompletion.

## The Problem Instructor Solves

Extracting structured data from LLMs traditionally involves:

1.  Writing and maintaining complex JSON schemas.
2.  Handling validation errors.
3.  Implementing retry logic.
4.  Parsing unstructured responses.
5.  Adapting to different provider APIs.

**Instructor streamlines this process with a simple interface.**

| Without Instructor                                                                                                                                                                                 | With Instructor                                                                                                                                                                                                     |
| -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python<br>response = openai.chat.completions.create(<br>    model="gpt-4",<br>    messages=[{"role": "user", "content": "..."}],<br>    tools=[<br>        {<br>            "type": "function",<br>            "function": {<br>                "name": "extract_user",<br>                "parameters": {<br>                    "type": "object",<br>                    "properties": {<br>                        "name": {"type": "string"},<br>                        "age": {"type": "integer"},<br>                    },<br>                },<br>            },<br>        }<br>    ],<br>)<br><br># Parse response<br>tool_call = response.choices[0].message.tool_calls[0]<br>user_data = json.loads(tool_call.function.arguments)<br><br># Validate manually<br>if "name" not in user_data:<br>    # Handle error...<br>    pass<br>``` | ```python<br>client = instructor.from_provider("openai/gpt-4")<br><br>user = client.chat.completions.create(<br>    response_model=User,<br>    messages=[{"role": "user", "content": "..."}],<br>)<br><br># That's it! user is validated and typed<br>``` |

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

Instructor offers a unified API across major LLM providers:

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

Instructor automatically retries failed extractions based on validation errors:

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

Stream partial object responses as they are generated:

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

## Trusted by Thousands

Instructor is a proven solution, trusted by over 100,000 developers and companies.

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Getting Started

### Basic Extraction

Easily extract structured data from any text:

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

### Multilingual Support

Instructor's simple API is available in multiple languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Choose Instructor?

**Compared to raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.

**Compared to LangChain/LlamaIndex:** Instructor focuses on structured extraction, offering a lighter, faster, and more debuggable solution.

**Compared to Custom Solutions:** Instructor is battle-tested by thousands of developers and handles edge cases effectively.

## Contributing

Contributions are welcome! Check out the [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - See [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>