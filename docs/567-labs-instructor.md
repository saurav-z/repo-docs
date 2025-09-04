# Instructor: Effortlessly Extract Structured Data from LLMs

**Stop wrestling with JSON parsing and error handling!** Instructor simplifies the process of extracting reliable, structured data from any Large Language Model (LLM) using Pydantic for seamless validation and type safety.  [Check out the original repo](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Define your desired output model using Pydantic, and Instructor handles the rest.
*   **Automatic Validation:** Ensures data integrity with built-in validation and error handling.
*   **Automatic Retries:** Automatically retries extractions upon validation failures.
*   **Streaming Support:**  Receive partial objects as they're generated for a more responsive user experience.
*   **Nested Object Support:** Easily extract complex, hierarchical data structures.
*   **Provider Agnostic:** Works seamlessly with all major LLM providers.

## The Challenge of LLM Data Extraction

Extracting structured data from LLMs is notoriously complex, involving:

*   Writing intricate JSON schemas.
*   Managing validation errors.
*   Implementing retry mechanisms.
*   Parsing unstructured responses.
*   Adapting to different API providers.

**Instructor simplifies this process dramatically:**

| Without Instructor                                                                                                                              | With Instructor                                                                                                                                                     |
| :--------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| ```python<br>response = openai.chat.completions.create(<br>    model="gpt-4",<br>    messages=[{"role": "user", "content": "..."}],<br>    tools=[<br>        {<br>            "type": "function",<br>            "function": {<br>                "name": "extract_user",<br>                "parameters": {<br>                    "type": "object",<br>                    "properties": {<br>                        "name": {"type": "string"},<br>                        "age": {"type": "integer"},<br>                    },<br>                },<br>            },<br>        }<br>    ],<br>)<br><br># Parse response<br>tool_call = response.choices[0].message.tool_calls[0]<br>user_data = json.loads(tool_call.function.arguments)<br><br># Validate manually<br>if "name" not in user_data:<br>    # Handle error...<br>    pass<br>``` | ```python<br>client = instructor.from_provider("openai/gpt-4")<br><br>user = client.chat.completions.create(<br>    response_model=User,<br>    messages=[{"role": "user", "content": "..."}],<br>)<br><br># That's it! user is validated and typed<br>``` |

## Installation

Get started in seconds:

```bash
pip install instructor
```

Or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## LLM Provider Compatibility

Use the same code across any major LLM provider:

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

Instructor automatically retries failed validations using the error message:

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

Effortlessly handle complex, nested data structures:

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

## Used in Production

Trusted by over 100,000 developers and companies, including teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Getting Started

### Basic Extraction

Extract structured data from unstructured text:

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

### Cross-Language Support

Instructor's API is available in multiple languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Instructor vs. Alternatives

**vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, without manual schema writing.

**vs LangChain/LlamaIndex:** Instructor is laser-focused on structured extraction, making it lighter, faster, and easier to debug.

**vs Custom Solutions:** Benefit from a battle-tested solution used by thousands of developers, handling edge cases you haven't encountered yet.

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>