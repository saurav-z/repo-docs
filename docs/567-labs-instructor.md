# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your AI development and get reliable, validated JSON outputs from any Large Language Model (LLM) with Instructor.**

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Effortless Data Extraction:** Get validated JSON data from LLMs without complex parsing or error handling.
*   **Pydantic Integration:** Leverage the power of Pydantic for type safety, validation, and IDE support.
*   **Automatic Retries:**  Handles failed extractions automatically, improving reliability.
*   **Streaming Support:**  Receive partial objects as they're generated.
*   **Nested Object Support:** Extract complex, nested data structures with ease.
*   **Broad Provider Compatibility:** Works seamlessly with major LLM providers (OpenAI, Anthropic, Google, Ollama, and more).

## The Problem Instructor Solves

Getting structured data from LLMs typically involves a lot of manual work:

1.  Writing complex JSON schemas.
2.  Handling validation errors.
3.  Retrying failed extractions.
4.  Parsing unstructured responses.
5.  Dealing with different provider APIs.

**Instructor simplifies this process with a single, intuitive interface:**

| **Without Instructor**                                                                                                                             | **With Instructor**                                                                                                     |
| :--------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------------- |
| ```python  response = openai.chat.completions.create(model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[{ "type": "function",  "function": {  "name": "extract_user",  "parameters": { "type": "object", "properties": { "name": {"type": "string"}, "age": {"type": "integer"},  },  },  } } ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments)  # Validate manually if "name" not in user_data:  # Handle error... pass ``` | ```python  client = instructor.from_provider("openai/gpt-4")  user = client.chat.completions.create(  response_model=User,  messages=[{"role": "user", "content": "..."}],  )  # That's it! user is validated and typed ``` |

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

## Comprehensive LLM Provider Support

Instructor is compatible with a wide range of LLM providers, allowing you to use the same code regardless of your chosen model:

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

Instructor automatically retries when validation fails, using the error message for improved accuracy:

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

Easily extract and handle complex, nested data structures:

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

Instructor is a battle-tested library used by over 100,000 developers and companies to build AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started Quickly

### Basic Data Extraction

Extract structured data from any text with ease:

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

*   **Superior to Raw JSON Mode:** Provides automatic validation, retries, streaming, and nested object support – without manual schema creation.
*   **More Focused than LangChain/LlamaIndex:**  Instructor focuses on structured extraction, offering a lighter, faster, and easier-to-debug solution.
*   **Battle-Tested & Robust:** Used by thousands of developers, Instructor handles edge cases you may not have considered.

## Contributing

We welcome contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>