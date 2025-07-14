# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON with Instructor – the Python library built on Pydantic for robust, type-safe, and production-ready structured output.** ([View original repository](https://github.com/567-labs/instructor))

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Eliminate complex JSON schema writing and manual parsing.
*   **Built-in Validation:** Leverage Pydantic for robust data validation, type safety, and IDE support.
*   **Automatic Retries:** Handles LLM failures gracefully with automatic retry mechanisms.
*   **Streaming Support:** Receive partial objects in real-time as the LLM generates them.
*   **Nested Object Support:** Seamlessly extract complex, nested data structures.
*   **Provider Agnostic:** Works with leading LLM providers like OpenAI, Anthropic, Google, and Ollama.

## The Problem Instructor Solves

Getting structured data from LLMs can be a cumbersome process. It typically involves:

1.  Writing intricate JSON schemas.
2.  Managing and handling validation errors.
3.  Implementing retry mechanisms for failed extractions.
4.  Parsing unstructured responses and handling various provider APIs.

**Instructor streamlines this process with a single, intuitive interface.**

| **Without Instructor**                                                                                                                                                                                                                                                                     | **With Instructor**                                                                                                       |
| :--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------- |
| ```python  response = openai.chat.completions.create(model="gpt-4",  messages=[{"role": "user", "content": "..."}],  tools=[  {"type": "function", "function": { "name": "extract_user",  "parameters": {  "type": "object",  "properties": {  "name": {"type": "string"},  "age": {"type": "integer"},  },  },  },  } ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0]  user_data = json.loads(tool_call.function.arguments)  # Validate manually if "name" not in user_data:  # Handle error...  pass ``` | ```python  client = instructor.from_provider("openai/gpt-4")  user = client.chat.completions.create(  response_model=User,  messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Installation

Install Instructor in seconds using pip or your preferred package manager:

```bash
pip install instructor
```

## Supported LLM Providers

Instructor is compatible with a wide range of LLM providers, allowing you to use the same code across different models:

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

## Production-Ready Features

### Automatic Retries

Instructor automatically retries failed validations, incorporating the error message to improve extraction success.

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

Stream partial objects as they're generated for a more interactive experience:

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

Effortlessly extract complex data structures with nested objects support.

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

## Trusted by Developers Worldwide

Instructor is a proven solution, trusted by a large and active community:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Get Started

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

### Multiple Languages

Instructor's simple API is available in many languages, expanding its usability.

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

**vs Raw JSON Mode**: Instructor provides automatic validation, retries, streaming, and nested object support. Eliminates the need for manual schema writing.

**vs LangChain/LlamaIndex**: Instructor is laser-focused on structured extraction. It's lighter, faster, and simplifies debugging.

**vs Custom Solutions**: Benefit from a battle-tested library, used by thousands of developers. Handles edge cases, saving time and effort.

## Contributing

We welcome contributions! Explore [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>