# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for type safety and ease of use.**  [See the original repo](https://github.com/567-labs/instructor)

Instructor empowers developers to seamlessly extract structured data from any Large Language Model (LLM), eliminating the complexities of manual parsing, validation, and error handling.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Define data models using Pydantic and get structured output effortlessly.
*   **Automatic Validation:** Built-in validation using Pydantic ensures data integrity and type safety.
*   **Provider Agnostic:** Works seamlessly with leading LLM providers like OpenAI, Anthropic, Google, and Ollama.
*   **Automatic Retries:** Handles validation failures with automatic retries and error reporting.
*   **Streaming Support:** Stream partial objects as they are generated for a responsive user experience.
*   **Nested Object Support:** Easily extract complex, nested data structures.

## Why Choose Instructor?

Instructor streamlines the process of obtaining structured data from LLMs, eliminating the need for manual work and reducing the risk of errors. Here's a comparison:

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

or with your package manager:

```bash
uv add instructor
poetry add instructor
```

## Example Usage

### Basic Extraction

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

## Production-Ready Features

### Automatic Retries

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

## Multi-Language Support

Instructor's simple API is available in many languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

## Used in Production by

Instructor is a trusted solution used by over 100,000 developers and companies, including teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why use Instructor?

**Instructor vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.

**Instructor vs LangChain/LlamaIndex**: Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.

**Instructor vs Custom solutions**: Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>