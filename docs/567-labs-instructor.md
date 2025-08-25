# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable JSON outputs with Instructor, the Python library built for structured data extraction.** [(View the original repository)](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simple API:** Define your data model using Pydantic and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity with built-in validation, type safety, and IDE support.
*   **Automatic Retries:** Handles extraction failures with intelligent retries and error messages.
*   **Streaming Support:** Get partial objects as they're generated, enabling real-time data processing.
*   **Nested Object Support:** Seamlessly extracts complex, nested data structures.
*   **Multi-Provider Compatibility:** Works with leading LLM providers like OpenAI, Anthropic, Google, and Ollama.

## Why Instructor?

Extracting structured data from LLMs can be complex. Instructor streamlines the process, eliminating the need for:

*   Writing intricate JSON schemas.
*   Manually handling validation errors.
*   Implementing retry logic.
*   Parsing unstructured responses.
*   Adapting to different provider APIs.

**Instructor offers a clean and efficient solution:**

| **Without Instructor**                                                        | **With Instructor**                                                          |
| :---------------------------------------------------------------------------- | :--------------------------------------------------------------------------- |
| ```python                                                                    | ```python                                                                   |
| response = openai.chat.completions.create(                                     | client = instructor.from_provider("openai/gpt-4")                            |
|     model="gpt-4",                                                              | user = client.chat.completions.create(                                       |
|     messages=[{"role": "user", "content": "..."}],                                |     response_model=User,                                                       |
|     tools=[                                                                     |     messages=[{"role": "user", "content": "..."}],                             |
|         {"type": "function", "function": {"name": "extract_user", "parameters": {"type": "object", "properties": {"name": {"type": "string"}, "age": {"type": "integer"}},}}}  | )                                                                          |
|     ],                                                                        |                                                                              |
| )                                                                             | # That's it! user is validated and typed                                      |
| tool_call = response.choices[0].message.tool_calls[0]                          |                                                                              |
| user_data = json.loads(tool_call.function.arguments)                          |                                                                              |
| if "name" not in user_data:                                                    |                                                                              |
|     pass  # Handle error...                                                    |                                                                              |
| ```                                                                         | ```                                                                          |

## Get Started in Seconds

```bash
pip install instructor
```

Or with your package manager:

```bash
uv add instructor
poetry add instructor
```

## Seamless Integration with Major LLM Providers

Instructor offers a consistent API across various LLM providers:

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

Instructor automatically retries failed validations, incorporating error messages for improved accuracy:

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

Stream partial objects in real time:

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

Trusted by over 100,000 developers and companies building AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Start Building Today

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

### Multilingual Support

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

## Instructor vs. Alternatives

**vs Raw JSON Mode**: Instructor provides automated validation, retries, streaming, and nested object support, eliminating the need for manual schema creation.

**vs LangChain/LlamaIndex**: Instructor is laser-focused on structured extraction, resulting in a lighter, faster, and more debuggable solution.

**vs Custom Solutions**: Benefit from a battle-tested solution used by thousands of developers, addressing complex edge cases.

## Contribute

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>