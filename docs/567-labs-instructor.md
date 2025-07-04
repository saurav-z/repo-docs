# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and reliably extract structured data with Instructor, built on Pydantic for validation and type safety.**  [View the original repository](https://github.com/567-labs/instructor)

![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)
![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)
![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)
![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)
![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)

## Key Features

*   **Simplified Data Extraction:** Define your desired output format with Pydantic models and let Instructor handle the rest.
*   **Automatic Validation:** Built-in Pydantic integration ensures data integrity and type safety.
*   **Intelligent Retries:** Automatically retries extraction attempts with error messages for robust results.
*   **Streaming Support:**  Receive partial objects as they're generated for real-time updates.
*   **Nested Object Support:** Easily handle complex, nested data structures.
*   **Provider Agnostic:** Works seamlessly with leading LLM providers like OpenAI, Anthropic, Google, and local models.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust.

## The Problem: Extracting Structured Data from LLMs is Hard

Traditional methods for extracting structured data from LLMs are complex and time-consuming. They involve:

*   Writing and managing complex JSON schemas.
*   Handling validation errors manually.
*   Implementing retry mechanisms.
*   Parsing unstructured responses.
*   Adapting to different provider APIs.

**Instructor eliminates these challenges with a simple, elegant solution.**

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

## Get Started in Seconds

```bash
pip install instructor
```

Or use your preferred package manager:
```bash
uv add instructor
poetry add instructor
```

## Seamless Integration with Top LLM Providers

Use the same Instructor code across all major LLM providers:

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

## Production-Ready Features

### Automatic Retries with Error Handling

Instructor automatically retries failed extractions based on validation errors:

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

Stream partial objects as they are generated for improved user experience:

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

Effortlessly extract complex, nested data structures:

```python
from typing import List
from pydantic import BaseModel

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

Instructor is a proven solution, trusted by over 100,000 developers and companies building AI applications.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and numerous YC startups.

## Quick Start: Examples

### Basic Data Extraction

Extract structured data from any text input:

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

## Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Choose Instructor?

*   **vs Raw JSON mode:** Instructor automates validation, retries, and supports streaming and nested objects, eliminating the need for manual schema creation.
*   **vs LangChain/LlamaIndex:** Instructor focuses solely on structured extraction, making it lightweight, fast, and easier to debug.
*   **vs Custom Solutions:** Benefit from a battle-tested solution used by thousands of developers, handling complex edge cases.

## Contribute

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>