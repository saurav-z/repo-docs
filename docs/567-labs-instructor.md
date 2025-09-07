# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your interactions with Large Language Models (LLMs) by effortlessly extracting reliable, validated JSON data using Instructor, built on Pydantic.**

[Go to the original repository](https://github.com/567-labs/instructor)

<p align="center">
  <a href="https://pypi.org/project/instructor/">
    <img src="https://img.shields.io/pypi/v/instructor?style=flat-square" alt="PyPI Version">
  </a>
  <a href="https://pypi.org/project/instructor/">
    <img src="https://img.shields.io/pypi/dm/instructor?style=flat-square" alt="PyPI Downloads">
  </a>
  <a href="https://github.com/instructor-ai/instructor">
    <img src="https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square" alt="GitHub Stars">
  </a>
  <a href="https://discord.gg/bD9YE9JArw">
    <img src="https://img.shields.io/discord/1192334452110659664?style=flat-square" alt="Discord">
  </a>
  <a href="https://twitter.com/jxnlco">
    <img src="https://img.shields.io/twitter/follow/jxnlco?style=flat-square" alt="Twitter">
  </a>
</p>

## Key Features of Instructor

*   **Simplified Data Extraction:** Eliminate complex JSON parsing and error handling.
*   **Pydantic Integration:** Leverage Pydantic for robust validation, type safety, and IDE support.
*   **Automatic Retries:** Handle LLM output failures with automatic retries based on validation errors.
*   **Streaming Support:** Stream partial objects for real-time data processing.
*   **Nested Object Support:** Extract complex, nested data structures effortlessly.
*   **Provider Agnostic:** Works seamlessly with various LLM providers, including OpenAI, Anthropic, Google, and local models.
*   **Production Ready**: Battle tested by thousands of developers.

## The Problem Instructor Solves

Extracting structured data from LLMs traditionally involves:

*   Writing intricate JSON schemas.
*   Dealing with validation errors.
*   Implementing retry mechanisms.
*   Parsing unstructured responses.
*   Adapting to different provider APIs.

**Instructor streamlines this process with a simple and intuitive interface:**

| Without Instructor                                                                                                                                                                                                | With Instructor                                                                                                                              |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python<br>response = openai.chat.completions.create(<br>    model="gpt-4",<br>    messages=[{"role": "user", "content": "..."}],<br>    tools=[<br>        {<br>            "type": "function",<br>            "function": {<br>                "name": "extract_user",<br>                "parameters": {<br>                    "type": "object",<br>                    "properties": {<br>                        "name": {"type": "string"},<br>                        "age": {"type": "integer"},<br>                    },<br>                },<br>            },<br>        }<br>    ],<br>)<br><br># Parse response<br>tool_call = response.choices[0].message.tool_calls[0]<br>user_data = json.loads(tool_call.function.arguments)<br><br># Validate manually<br>if "name" not in user_data:<br>    # Handle error...<br>    pass<br>``` | ```python<br>client = instructor.from_provider("openai/gpt-4")<br><br>user = client.chat.completions.create(<br>    response_model=User,<br>    messages=[{"role": "user", "content": "..."}],<br>)<br><br># That's it! user is validated and typed<br>``` |

## Installation

Get started in seconds:

```bash
pip install instructor
```

or, using your preferred package manager:
```bash
uv add instructor
poetry add instructor
```

## LLM Provider Compatibility

Instructor works with a wide range of LLM providers:

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

## Advanced Features for Production

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

Receive partial objects as they are generated:

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

Easily extract complex, nested data:

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

## Trusted by Developers

Instructor is used by:

*   **100,000+ Developers & Companies**
*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups utilize Instructor.

## Get Started

### Basic Extraction

Extract structured data quickly:

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

Instructor offers simple APIs in multiple languages:

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

## Instructor vs. Alternatives

**vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.

**vs LangChain/LlamaIndex:** Instructor specializes in structured extraction, offering a lighter, faster, and easier-to-debug solution.

**vs Custom solutions:** Instructor is battle-tested by thousands of developers and handles complex edge cases.

## Contributing

Contributions are welcome! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>