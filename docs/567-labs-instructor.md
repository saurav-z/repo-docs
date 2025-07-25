# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, structured data with Instructor, built on Pydantic for validation, type safety, and seamless integration.**  Check out the original repo: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Effortless Data Extraction:** Define your desired data structure using Pydantic models and let Instructor handle the rest.
*   **Provider Agnostic:** Works seamlessly with leading LLM providers like OpenAI, Anthropic, Google, and more.
*   **Automatic Retries:** Built-in retry mechanism to handle validation errors, ensuring reliable results.
*   **Streaming Support:** Receive partial objects in real-time as the LLM generates data.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Type Safety and Validation:** Leverage Pydantic for robust data validation and type safety.
*   **Cross-Platform Support:**  Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## Why Use Instructor?

Instructor simplifies the process of extracting structured data from LLMs, eliminating the complexities of:

*   Writing complex JSON schemas
*   Handling validation errors
*   Retrying failed extractions
*   Parsing unstructured responses
*   Dealing with different provider APIs

| **Without Instructor** | **With Instructor** |
| --------------------- | ------------------- |
| ```python<br>response = openai.chat.completions.create(<br>    model="gpt-4",<br>    messages=[{"role": "user", "content": "..."}],<br>    tools=[<br>        {<br>            "type": "function",<br>            "function": {<br>                "name": "extract_user",<br>                "parameters": {<br>                    "type": "object",<br>                    "properties": {<br>                        "name": {"type": "string"},<br>                        "age": {"type": "integer"},<br>                    },<br>                },<br>            },<br>        }<br>    ],<br>)<br><br># Parse response<br>tool_call = response.choices[0].message.tool_calls[0]<br>user_data = json.loads(tool_call.function.arguments)<br><br># Validate manually<br>if "name" not in user_data:<br>    # Handle error...<br>    pass``` | ```python<br>client = instructor.from_provider("openai/gpt-4")<br><br>user = client.chat.completions.create(<br>    response_model=User,<br>    messages=[{"role": "user", "content": "..."}],<br>)<br><br># That's it! user is validated and typed``` |

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

## Provider Support

Instructor is compatible with various LLM providers, enabling you to use the same code across different platforms:

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

Instructor automatically retries when validation fails, ensuring robust and reliable data extraction:

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

Stream partial objects as they are generated:

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

Easily extract complex, nested data structures:

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

Instructor is a proven solution used in production by over 100,000 developers and companies, with impressive community support.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Get Started

### Basic Extraction

Extract structured data with a simple implementation:

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

Instructor's API is available in multiple languages, enabling you to choose the best fit for your project:

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

## Why Use Instructor Over Alternatives?

**vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.

**vs LangChain/LlamaIndex**: Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.

**vs Custom solutions**: Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contributing

We encourage contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>