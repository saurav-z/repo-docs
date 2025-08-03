# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for type safety and ease of use.**  [View the original repository](https://github.com/567-labs/instructor).

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Seamless Data Extraction:**  Get structured data (like JSON) directly from LLMs without manual parsing.
*   **Pydantic Integration:** Leverage Pydantic for robust validation, type safety, and IDE support.
*   **Automatic Error Handling:** Built-in retries for failed extractions, ensuring reliable results.
*   **Provider Agnostic:** Works with all major LLM providers, including OpenAI, Anthropic, Google, and local models (Ollama).
*   **Streaming Support:**  Receive and process partial data as it's generated.
*   **Nested Object Support:** Easily handle complex, nested data structures.
*   **Production Ready:** Trusted by over 100,000 developers and companies.

## The Problem Instructor Solves

Extracting structured data from LLMs is often complex and time-consuming, requiring:

*   Writing and maintaining complex JSON schemas
*   Handling validation errors and retries
*   Parsing and validating unstructured responses
*   Adapting to different LLM provider APIs

**Instructor simplifies this with a simple, intuitive interface.**

| **Without Instructor**  | **With Instructor** |
|------------------------|---------------------|
|  ```python <br> response = openai.chat.completions.create(<br>  model="gpt-4",<br>  messages=[{"role": "user", "content": "..."}],<br>  tools=[<br>   {<br>    "type": "function",<br>    "function": {<br>     "name": "extract_user",<br>     "parameters": {<br>      "type": "object",<br>      "properties": {<br>       "name": {"type": "string"},<br>       "age": {"type": "integer"},<br>      },<br>     },<br>    },<br>   }<br>  ],<br> )<br>  # Parse response<br>  tool_call = response.choices[0].message.tool_calls[0]<br>  user_data = json.loads(tool_call.function.arguments)<br>  # Validate manually<br>  if "name" not in user_data:<br>   # Handle error...<br>   pass  ```  |  ```python <br>  client = instructor.from_provider("openai/gpt-4")<br>  user = client.chat.completions.create(<br>   response_model=User,<br>   messages=[{"role": "user", "content": "..."}],<br>  )<br>  # That's it! user is validated and typed  ``` |

## Installation

Install Instructor in seconds using pip or your preferred package manager:

```bash
pip install instructor
```

Or using:

```bash
uv add instructor
poetry add instructor
```

## Provider Compatibility

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

Instructor automatically retries failed validations based on error messages, improving reliability:

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

Stream partial object responses as they're generated:

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

##  Used In Production By

Instructor is a battle-tested library that is trusted by over 100,000 developers and companies building AI applications.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Getting Started

### Basic Extraction Example

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

### Multi-language Support

Instructor's simple API is available in many languages:

*   [Python](https://python.useinstructor.com) - The original
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

## Instructor vs. Alternatives

**vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support, eliminating the need for manual schema creation.

**vs LangChain/LlamaIndex:** Instructor is focused on a single, core task – structured extraction, making it lightweight, fast, and easier to debug.

**vs Custom solutions:** Instructor is battle-tested by thousands of developers and handles many edge cases.

## Contributing

We welcome contributions!  Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>