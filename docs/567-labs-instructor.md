# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM integrations and get reliable JSON with Instructor, the Python library built for seamless data extraction and backed by Pydantic's power.**

[Go to the Original Repo](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Effortless Data Extraction:** Eliminate manual JSON parsing, error handling, and retries.
*   **Pydantic Integration:** Leverage Pydantic for type safety, validation, and IDE support.
*   **Automatic Retries:** Built-in retry mechanism for handling failed validations.
*   **Streaming Support:** Receive partial objects as they are generated.
*   **Nested Object Support:** Easily extract complex, nested data structures.
*   **Provider Agnostic:** Works with OpenAI, Anthropic, Google, Ollama, and more, using a unified API.

## The Challenge of Structured Data Extraction

Extracting structured data from LLMs is often a complex and time-consuming process, requiring:

1.  Writing complex JSON schemas
2.  Handling validation errors
3.  Retrying failed extractions
4.  Parsing unstructured responses
5.  Dealing with different provider APIs

**Instructor simplifies this with a single, intuitive interface:**

<table>
<tr>
<td><b>Without Instructor</b></td>
<td><b>With Instructor</b></td>
</tr>
<tr>
<td>

```python
response = openai.chat.completions.create(
    model="gpt-4",
    messages=[{"role": "user", "content": "..."}],
    tools=[
        {
            "type": "function",
            "function": {
                "name": "extract_user",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "age": {"type": "integer"},
                    },
                },
            },
        }
    ],
)

# Parse response
tool_call = response.choices[0].message.tool_calls[0]
user_data = json.loads(tool_call.function.arguments)

# Validate manually
if "name" not in user_data:
    # Handle error...
    pass
```

</td>
<td>

```python
client = instructor.from_provider("openai/gpt-4")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)

# That's it! user is validated and typed
```

</td>
</tr>
</table>

## Installation

Get started in seconds:

```bash
pip install instructor
```

Or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Seamless Provider Integration

Use the same code with any major LLM provider:

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

## Production-Ready Capabilities

*   **Automatic Retries:** Instructor automatically retries failed extractions, streamlining your workflow.
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

    user = client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "..."}],
        max_retries=3,
    )
    ```
*   **Streaming Support:** Stream partial objects as they are generated, enhancing real-time applications.
    ```python
    from instructor import Partial

    for partial_user in client.chat.completions.create(
        response_model=Partial[User],
        messages=[{"role": "user", "content": "..."}],
        stream=True,
    ):
        print(partial_user)
    ```
*   **Nested Objects:** Easily handle complex, nested data structures, simplifying advanced extraction tasks.
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

    user = client.chat.completions.create(
        response_model=User,
        messages=[{"role": "user", "content": "..."}],
    )
    ```

## Trusted by Thousands

Instructor is used by over 100,000 developers and companies:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Quick Start

### Basic Extraction

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

Instructor's API is available in multiple languages for broader accessibility:

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

## Why Choose Instructor?

*   **Compared to Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating the need for manual schema writing.
*   **Compared to LangChain/LlamaIndex:** Instructor offers a focused solution, streamlining structured extraction for a lighter, faster, and more debuggable experience.
*   **Compared to Custom Solutions:** Leverage a battle-tested library trusted by thousands of developers, handling edge cases efficiently.

## Contributing

Contributions are welcome! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>