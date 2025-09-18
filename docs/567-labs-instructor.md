# Instructor: Easily Extract Structured Data from LLMs

**Simplify your AI workflows and get reliable JSON output from any Large Language Model (LLM) with Instructor, built on Pydantic for robust validation, type safety, and streamlined development.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Eliminate manual JSON parsing and error handling.
*   **Pydantic Integration:** Leverage the power of Pydantic for validation, type safety, and IDE support.
*   **Automatic Retries:** Automatically retry failed extractions based on validation errors.
*   **Streaming Support:** Stream partial objects as they're generated for faster feedback.
*   **Nested Object Support:** Effortlessly extract complex, nested data structures.
*   **Provider Agnostic:** Works with OpenAI, Anthropic, Google, Ollama, and more.
*   **Cross-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Instructor?

Instructor simplifies the process of getting structured data from LLMs by handling complex challenges:

*   **No More Complex Schemas:** Forget writing intricate JSON schemas.
*   **Handles Validation Errors:** Automatic error handling and retries for increased reliability.
*   **Parses Unstructured Responses:** Converts raw responses into structured data.
*   **Supports Multiple APIs:** Works seamlessly with various LLM providers.

**Instructor simplifies the process. Instead of:**

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "extract_user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        }
    ],
)

# Parse response
tool_call = response.choices[0].message.tool_calls[0]
user_data = json.loads(tool_call.function.arguments)

# Validate manually
if "name" not in user_data:
    # Handle error...
    pass
```

**You can simply use:**

```python
client = instructor.from_provider("openai/gpt-4")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)

# That's it! user is validated and typed
```

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

## LLM Provider Compatibility

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

*   **Automatic Retries:** Instructor automatically retries failed validations.

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

*   **Streaming Support:** Stream partial objects as they are generated.

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

*   **Nested Objects:** Extract complex nested data structures.

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

## Trusted by the Community

Instructor is used in production by:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Trusted by developers and companies at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

### Basic Extraction

Extract structured data from text:

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

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Use Instructor?

*   **vs Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support.
*   **vs LangChain/LlamaIndex:** Instructor is focused on structured extraction, making it lighter, faster, and easier to debug.
*   **vs Custom Solutions:** Instructor is battle-tested and handles edge cases.

## Contributing

Contributions are welcome! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>