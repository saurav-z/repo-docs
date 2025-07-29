# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, structured JSON outputs with Instructor, the Python library built on Pydantic for type safety and ease of use.**  [Check out the original repo!](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Effortless Extraction:** Define your desired output model, and Instructor handles the rest.
*   **Pydantic Integration:** Leverages Pydantic for robust validation, type safety, and IDE support.
*   **Automatic Retries:** Handles extraction failures automatically, improving reliability.
*   **Streaming Support:** Get partial results in real-time as they are generated.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Multiple Provider Support:** Works seamlessly with leading LLM providers like OpenAI, Anthropic, Google, and local models via Ollama.

## Why Choose Instructor?

Manually parsing and validating data from LLMs is complex and error-prone. Instructor simplifies this process by:

*   Eliminating the need for complex JSON schemas.
*   Automating validation error handling.
*   Handling retry mechanisms for failed extractions.
*   Parsing unstructured responses and returning structured data.
*   Providing a unified API across various LLM providers.

### Without Instructor vs. With Instructor: A Comparison

| Without Instructor                                                                                                                                                                                            | With Instructor                                                                                                                                                                                                                                     |
| ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[ { "type": "function", "function": { "name": "extract_user", "parameters": { ... } } } ], ) # Parse response # Validate manually ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Getting Started

### Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Extraction Example

Easily extract structured data:

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

## Production-Ready Capabilities

### Automatic Retries

Instructor automatically retries failed extractions, improving reliability:

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

## Comprehensive Provider Support

Instructor seamlessly integrates with all major LLM providers, enabling you to write code once and use it everywhere.

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

## Used in Production

Instructor is trusted by a large and growing community of developers and companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Join the ranks of over 100,000 developers and companies including teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Instructor in Multiple Languages

Instructor's simple API is available in several languages, expanding accessibility for a broader audience:

*   [Python](https://python.useinstructor.com) - The Original
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

**Instructor vs. Alternatives**

*   **Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.
*   **LangChain/LlamaIndex:** Instructor is laser-focused on structured extraction. It's lighter, faster, and easier to debug.
*   **Custom Solutions:** Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>