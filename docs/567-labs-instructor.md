# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic for type safety and easy integration.**  ([See the original repo](https://github.com/567-labs/instructor))

Instructor eliminates the complexities of extracting structured data from Large Language Models (LLMs), offering a streamlined and robust solution.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   ✅ **Simplified Data Extraction:** Define Python models (using Pydantic) and get structured data directly from LLMs.
*   ✅ **Automatic Validation:**  Ensures data integrity with built-in validation and type checking.
*   ✅ **Error Handling & Retries:** Automatically handles common LLM response errors and retries for reliable results.
*   ✅ **Seamless Provider Integration:** Works with all major LLM providers like OpenAI, Anthropic, Google, and local models (Ollama).
*   ✅ **Streaming Support:** Receive partial responses in real-time as they are generated.
*   ✅ **Nested Object Support:**  Effortlessly handle complex, nested data structures.

## Why Choose Instructor?

Instructor simplifies LLM interactions, providing:

*   **No Complex Schema Creation:** Avoid writing intricate JSON schemas.
*   **No Manual Validation:**  Eliminate the need for manual error handling.
*   **No Frustrating Retries:**  Let Instructor handle failed extractions automatically.
*   **No Parsing Headaches:**  Focus on your data, not the extraction process.
*   **Consistent APIs:** Interact with various LLM providers using the same intuitive API.

**See the Difference:**

| **Without Instructor** | **With Instructor** |
|------------------------|---------------------|
| ```python response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[ { "type": "function", "function": { "name": "extract_user", "parameters": { "type": "object", "properties": { "name": {"type": "string"}, "age": {"type": "integer"}, }, }, }, } ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Installation

Get started in seconds with:

```bash
pip install instructor
```

Alternative package managers:

```bash
uv add instructor
poetry add instructor
```

## Provider Compatibility

Use the same code across multiple LLM providers:

```python
# OpenAI
client = instructor.from_provider("openai/gpt-4o")

# Anthropic
client = instructor.from_provider("anthropic/claude-3-5-sonnet")

# Google
client = instructor.from_provider("google/gemini-pro")

# Ollama (local)
client = instructor.from_provider("ollama/llama3.2")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-Ready Features

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

*   **Streaming Support:** Stream partial objects as they're generated:

    ```python
    from instructor import Partial

    for partial_user in client.chat.completions.create(
        response_model=Partial[User],
        messages=[{"role": "user", "content": "..."}],
        stream=True,
    ):
        print(partial_user)
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

## Trusted by a Large Community

Instructor is a popular choice for building AI applications:

*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1000+ Community Contributors**

Used in production by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

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

### Cross-Platform Compatibility

Instructor's easy-to-use API is available in many languages:

*   [Python](https://python.useinstructor.com)
*   [TypeScript](https://js.useinstructor.com)
*   [Ruby](https://ruby.useinstructor.com)
*   [Go](https://go.useinstructor.com)
*   [Elixir](https://hex.pm/packages/instructor)
*   [Rust](https://rust.useinstructor.com)

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Instructor vs. Alternatives

*   **vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing required.
*   **vs LangChain/LlamaIndex:** Instructor is focused on structured extraction, making it lighter, faster, and easier to debug.
*   **vs Custom solutions:** Instructor has been battle-tested by thousands of developers and handles edge cases effectively.

## Contribute

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get involved.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>