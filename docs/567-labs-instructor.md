# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM integrations and get reliable JSON outputs with Instructor, built on Pydantic for seamless validation and type safety.**  (See the original repo: [https://github.com/567-labs/instructor](https://github.com/567-labs/instructor))

Instructor empowers you to extract structured data from any language model with ease, handling complex tasks like JSON parsing, validation, retries, and streaming.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   ✅ **Simplified Extraction:** Define your desired data structure with Pydantic models and let Instructor handle the rest.
*   ✅ **Automatic Validation:** Ensures data integrity with built-in validation based on your Pydantic models.
*   ✅ **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   ✅ **Streaming Support:** Stream partial objects as they're generated for real-time data processing.
*   ✅ **Nested Object Support:** Easily extract complex, nested data structures.
*   ✅ **Cross-Provider Compatibility:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and local models (Ollama).
*   ✅ **Production-Ready:** Used by thousands of developers and companies, including teams at OpenAI, Google, Microsoft, and AWS.

## The Problem Instructor Solves

Extracting structured data from LLMs can be a complex and error-prone process, often involving:

*   Writing complex JSON schemas
*   Handling validation errors
*   Implementing retry logic
*   Parsing unstructured responses
*   Adapting to different provider APIs

**Instructor streamlines this process with a single, intuitive interface:**

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

## Getting Started: Install and Use

### Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or use your preferred package manager:
```bash
uv add instructor
poetry add instructor
```

### Basic Extraction Example

Define your data model and extract data effortlessly:

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

## Powerful Features for Production

*   **Automatic Retries:** Instructor automatically retries failed validations:

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
*   **Streaming Support:**  Stream partial objects as they're generated:

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

*   **Nested Objects:** Extract complex, nested data structures:

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

## Provider Compatibility

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

# With API keys directly
client = instructor.from_provider("openai/gpt-4o", api_key="sk-...")
client = instructor.from_provider("anthropic/claude-3-5-sonnet", api_key="sk-ant-...")
client = instructor.from_provider("groq/llama-3.1-8b-instant", api_key="gsk_...")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Used in Production By: Proven Results

Trusted by over 100,000 developers and companies:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Instructor is used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Multi-Language Support

Instructor's simple API is available in many languages:

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

## Instructor vs. Alternatives

**Why choose Instructor?**

*   **vs Raw JSON mode:**  Instructor offers automatic validation, retries, streaming, and nested object support. Eliminates the need for manual schema writing.
*   **vs LangChain/LlamaIndex:** Instructor focuses on structured extraction, providing a lighter, faster, and easier-to-debug solution.
*   **vs Custom Solutions:**  Instructor is battle-tested, handling edge cases and saving you valuable development time.

## Contribute

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>