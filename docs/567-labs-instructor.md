# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, validated JSON outputs with Instructor, a Python library built for developers.**  [Explore the original repo on GitHub](https://github.com/567-labs/instructor).

![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)
![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)
![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)
![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)

## Key Features

*   **Simplified Extraction:** Define data models using Pydantic and get structured data directly from LLMs.
*   **Automatic Validation:** Built-in validation to ensure data quality and prevent errors.
*   **Provider Agnostic:** Works seamlessly with OpenAI, Anthropic, Google, Ollama, and more.
*   **Automatic Retries:** Handles failed extractions with automatic retries for increased reliability.
*   **Streaming Support:** Receive partial objects as they're generated for real-time updates.
*   **Nested Objects:** Easily extract complex, nested data structures.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Instructor?

Dealing with structured data extraction from Large Language Models can be complex. Instructor streamlines this process by:

*   Eliminating the need for complex JSON schemas.
*   Handling validation errors and retries automatically.
*   Parsing unstructured responses efficiently.
*   Providing a unified interface across different LLM providers.

**Here's how Instructor simplifies your code:**

| **Without Instructor**                                                                                                                                                                                                                                                                                                                                | **With Instructor**                                                                                                                                                                                                                                                                                                                        |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python                                                                                                                                                                                                                                                                                                                                            | ```python                                                                                                                                                                                                                                                                                                                              |
| ```python response = openai.chat.completions.create(    model="gpt-4",    messages=[{"role": "user", "content": "..."}],    tools=[        {            "type": "function",            "function": {                "name": "extract_user",                "parameters": {                    "type": "object",                    "properties": {                        "name": {"type": "string"},                        "age": {"type": "integer"},                    },                },            },        }    ],) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data:    # Handle error...    pass ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create(    response_model=User,    messages=[{"role": "user", "content": "..."}],) # That's it! user is validated and typed ``` |

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Comprehensive Provider Support

Instructor works with all major LLM providers, providing a unified API:

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

Instructor automatically retries failed validations:

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

## Trusted by the Community

Instructor is used in production by thousands of developers and companies, including teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

## Get Started

### Basic Extraction

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

### Multi-Language Support

Instructor's simple API is available in:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

### Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Why Use Instructor?

**Instructor vs. Alternatives:**

*   **Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing is required.
*   **LangChain/LlamaIndex:** Instructor is focused solely on structured extraction, making it lighter, faster, and easier to debug.
*   **Custom Solutions:** Instructor offers a battle-tested solution, handling edge cases and providing a robust foundation for your projects.

## Contributing

We welcome contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>