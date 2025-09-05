# Instructor: Effortlessly Extract Structured Data from LLMs

**Stop wrestling with JSON parsing and complex schemas; Instructor simplifies LLM output extraction, delivering reliable, validated data with ease.** ([Original Repository](https://github.com/567-labs/instructor))

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   ✅ **Simplified Extraction:** Get structured data without manual JSON parsing or complex schema creation.
*   ✅ **Pydantic Integration:** Leverage Pydantic for seamless validation, type safety, and IDE support.
*   ✅ **Automatic Retries:** Handle LLM output errors with built-in retry mechanisms.
*   ✅ **Streaming Support:** Receive partial objects as they are generated for enhanced responsiveness.
*   ✅ **Nested Object Extraction:** Effortlessly extract complex, nested data structures.
*   ✅ **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, and more) using the same API.

## Why Instructor?

Instructor streamlines the process of extracting structured data from Large Language Models, solving the common pain points of:

*   Complex JSON schema creation
*   Error handling and validation
*   Retry mechanisms
*   Unstructured response parsing
*   API differences across providers

**Instructor provides a clean and efficient solution, replacing multiple steps with a single, intuitive interface.**

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

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Examples

### Basic Data Extraction

Easily extract structured data from any text input:

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

### Streaming

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

## Production-Ready Capabilities

*   **Automatic Retries:** Built-in retry mechanisms for handling validation errors.
*   **Streaming Support:** Stream partial objects as they're generated.
*   **Nested Object Extraction:** Effortlessly handle complex, nested data structures.

## Used by Thousands

Instructor is trusted by a large and growing community:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

### Basic Extraction

Extract structured data from any text:

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

Instructor's simple API is available in many languages:

- [Python](https://python.useinstructor.com) - The original
- [TypeScript](https://js.useinstructor.com) - Full TypeScript support
- [Ruby](https://ruby.useinstructor.com) - Ruby implementation
- [Go](https://go.useinstructor.com) - Go implementation
- [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
- [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Choose Instructor?

*   **vs Raw JSON mode:** Offers automatic validation, retries, streaming, and nested object support, eliminating the need for manual schema creation.
*   **vs LangChain/LlamaIndex:** Focused solely on structured extraction, providing a lighter, faster, and easier-to-debug solution.
*   **vs Custom Solutions:**  A battle-tested, community-driven library that handles edge cases you haven't considered yet.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>