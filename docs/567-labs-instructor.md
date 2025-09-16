# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your AI workflows and get reliable JSON from any Large Language Model (LLM) with Instructor, built on Pydantic for robust validation and type safety.** 

[View the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Extraction:** Easily define your desired output using Pydantic models.
*   **Robust Validation:** Benefit from automatic validation, ensuring data integrity.
*   **Automatic Retries:** Handle LLM output errors with built-in retry mechanisms.
*   **Streaming Support:** Get partial objects as they are generated.
*   **Nested Object Support:** Extract complex, nested data structures effortlessly.
*   **Provider Agnostic:** Works seamlessly with major LLM providers.
*   **Multi-Language Support**: Available in Python, Typescript, Ruby, Go, Elixir, and Rust.

## The Problem: Painful LLM Output Handling

Extracting structured data from LLMs is notoriously complex, requiring:

1.  Complex JSON Schema Writing
2.  Manual Validation Error Handling
3.  Retry Logic for Failed Extractions
4.  Parsing of Unstructured Responses
5.  Adapting to different Provider APIs

## Instructor Solution: Simple, Reliable Extraction

**Instructor simplifies data extraction with a single, intuitive interface:**

| **Before Instructor**                                                                                                    | **After Instructor**                                                                                                           |
| :----------------------------------------------------------------------------------------------------------------------- | :----------------------------------------------------------------------------------------------------------------------------- |
| ```python  response = openai.chat.completions.create(  model="gpt-4",  messages=[{"role": "user", "content": "..."}],  tools=[  {  "type": "function",  "function": {  "name": "extract_user",  "parameters": {  "type": "object",  "properties": {  "name": {"type": "string"},  "age": {"type": "integer"},  },  },  },  } ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass ``` | ```python client = instructor.from_provider("openai/gpt-4")  user = client.chat.completions.create(  response_model=User,  messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Installation

Install Instructor in seconds:

```bash
pip install instructor
```

Or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## LLM Provider Compatibility

Use the same code across various LLM providers:

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

Stream partial objects in real-time:

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

Effortlessly handle complex data structures:

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

Instructor is a popular choice for building AI applications, with:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Getting Started

### Basic Extraction Example

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

Instructor's simple API is available in multiple languages:

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

## Why Choose Instructor?

*   **Compared to Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support. It eliminates the need for manual schema creation.
*   **Compared to LangChain/LlamaIndex:** Instructor is specifically focused on structured extraction, offering a lighter, faster, and more easily debuggable solution.
*   **Compared to Custom Solutions:** Instructor is battle-tested by thousands of developers, addressing edge cases and saving you valuable time.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>