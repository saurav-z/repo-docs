# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your AI development: Instructor lets you reliably extract structured JSON data from any Large Language Model (LLM) with ease.**

[Visit the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Simplified Extraction:** Get structured data from LLMs with a single line of code using a familiar Pythonic interface.
*   **Pydantic Integration:** Leverages Pydantic for robust validation, type safety, and seamless IDE support.
*   **Automatic Error Handling:** Includes automatic retries for failed extractions, saving you time and effort.
*   **Flexible Provider Support:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, etc.) with a consistent API.
*   **Production-Ready Features:** Supports automatic retries, streaming for partial objects, and nested object extraction.
*   **Multi-Language Support:** Available in Python (original), TypeScript, Ruby, Go, Elixir, and Rust.

## The Challenge of Structured Data Extraction

Extracting structured data from LLMs can be a complex process, often requiring:

1.  Writing intricate JSON schemas.
2.  Handling validation errors meticulously.
3.  Implementing retry mechanisms for failures.
4.  Parsing unstructured responses into usable formats.
5.  Adapting to different provider APIs.

**Instructor simplifies this, providing a streamlined solution that handles all these complexities with an intuitive interface.**

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

## Getting Started

### Installation

Install Instructor in seconds using your preferred package manager:

```bash
pip install instructor
```

Or with other package managers:
```bash
uv add instructor
poetry add instructor
```

## Key Features

### Automatic Retries

Instructor automatically retries extractions when validation fails, utilizing the error message:

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

## Real-World Usage

Trusted by over 100,000 developers and companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Instructor is used by teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

### Basic Example: Data Extraction

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

## Instructor in Multiple Languages

Explore the Instructor API across various languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

## Learn More

*   [Documentation](https://python.useinstructor.com) - Detailed guides
*   [Examples](https://python.useinstructor.com/examples/) - Ready-to-use recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Instructor vs. Alternatives

**Instructor vs. Raw JSON Mode:** Instructor provides automatic validation, retries, streaming, and nested object support without manual schema writing.

**Instructor vs. LangChain/LlamaIndex:** Instructor is focused on streamlined structured extraction, making it lighter, faster, and easier to debug.

**Instructor vs. Custom Solutions:** Benefit from a battle-tested solution used by thousands of developers. Instructor handles edge cases efficiently.

## Contribute

Contributions are welcome! Check out the [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to start.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>