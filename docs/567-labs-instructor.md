# Instructor: Simplify LLM Output with Structured Data Extraction

**Effortlessly extract reliable JSON from any LLM with Instructor, leveraging Pydantic for validation, type safety, and seamless integration.** ([View on GitHub](https://github.com/567-labs/instructor))

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Get structured data without manual JSON parsing or error handling.
*   **Pydantic Integration:** Leverage Pydantic for type safety, validation, and IDE support.
*   **Automatic Retries:** Handle LLM failures with automatic retries and error reporting.
*   **Streaming Support:** Receive partial objects in real-time with streaming capabilities.
*   **Nested Object Support:** Extract complex, nested data structures effortlessly.
*   **Multi-Provider Compatibility:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and local models via Ollama.
*   **Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## The Problem Instructor Solves

Building with LLMs often requires extracting structured data from unstructured text. This is a complex process. Instructor simplifies this by:

1.  **Eliminating JSON Schema Writing:** No need to manually define and maintain complex schemas.
2.  **Automating Validation:** Automatically validates the extracted data against your defined model.
3.  **Simplifying Error Handling:** Handles validation errors and retries extraction automatically.
4.  **Streamlining Provider Integration:** Works with all major LLM providers using a unified interface.

**Instructor drastically reduces the code required:**

| **Without Instructor**                                                                                                                                                                                                                                                                                                                                           | **With Instructor**                                                                                                                                                                                                                           |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python  response = openai.chat.completions.create(   model="gpt-4",   messages=[{"role": "user", "content": "..."}],   tools=[       {           "type": "function",           "function": {               "name": "extract_user",               "parameters": {                   "type": "object",                   "properties": {                       "name": {"type": "string"},                       "age": {"type": "integer"},                   },               },           },       }   ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data:     # Handle error...     pass  ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create(   response_model=User,   messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Installation

Install Instructor in seconds:

```bash
pip install instructor
```

Or using other package managers:

```bash
uv add instructor
poetry add instructor
```

## Supported LLM Providers

Instructor supports a wide range of LLM providers, enabling you to use the same code across different models and services.  Here's how easy it is to switch providers:

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

## Production-Ready Features in Detail

### Automatic Retries

Instructor automatically retries failed validations, making your applications more robust:

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

Stream partial objects as they're generated, providing a more responsive user experience:

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

## Used in Production By

Instructor is trusted by a large and growing community of developers and companies:

*   **100,000+ Developers & Companies:** Building AI applications with Instructor.
*   **3M+ Monthly Downloads:**  A testament to Instructor's ease of use and effectiveness.
*   **10K+ GitHub Stars:**  A strong indication of community adoption and support.
*   **1000+ Community Contributors:**  A thriving community that contributes to Instructor's ongoing development.

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

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

### Multiple Languages

Instructor's simple API is available in many languages:

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

## Why Instructor?

**Compared to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support, without manual schema writing.

**Compared to LangChain/LlamaIndex:** Instructor focuses on structured extraction, making it lighter, faster, and easier to debug.

**Compared to Custom Solutions:**  Instructor is battle-tested by thousands of developers and handles complex extraction scenarios.

## Contributing

We welcome contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>