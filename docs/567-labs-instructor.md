# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, validated JSON outputs with Instructor, built on Pydantic.**

[<img src="https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square" alt="GitHub Stars">](https://github.com/567-labs/instructor)
[<img src="https://img.shields.io/pypi/v/instructor?style=flat-square" alt="PyPI">](https://pypi.org/project/instructor/)
[<img src="https://img.shields.io/pypi/dm/instructor?style=flat-square" alt="Downloads">](https://pypi.org/project/instructor/)
[<img src="https://img.shields.io/discord/1192334452110659664?style=flat-square" alt="Discord">](https://discord.gg/bD9YE9JArw)
[<img src="https://img.shields.io/twitter/follow/jxnlco?style=flat-square" alt="Twitter">](https://twitter.com/jxnlco)

## Key Features:

*   ✅ **Simplified Data Extraction:** Define your data model using Pydantic and let Instructor handle the rest.
*   ✅ **Automatic Validation:** Ensures data integrity with built-in validation using Pydantic.
*   ✅ **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   ✅ **Streaming Support:** Get partial objects as they are generated, enhancing user experience.
*   ✅ **Nested Object Support:** Handles complex, nested data structures seamlessly.
*   ✅ **Multi-Provider Compatibility:** Works seamlessly with major LLM providers like OpenAI, Anthropic, Google, and Ollama.

## The Problem: Structured Data Extraction from LLMs is Difficult

Extracting structured data from Large Language Models (LLMs) can be a complex and error-prone process. You often face challenges such as:

*   Crafting intricate JSON schemas
*   Managing validation errors
*   Implementing retry mechanisms
*   Parsing unstructured responses
*   Adapting to different provider APIs

## The Solution: Instructor Simplifies Everything

Instructor provides a streamlined interface to solve these problems:

| **Without Instructor**                                    | **With Instructor**                                          |
| :-------------------------------------------------------- | :----------------------------------------------------------- |
| Requires manual JSON parsing, schema definition, and error handling. | Define a Pydantic model and let Instructor handle the extraction. |
|                                                           | Data is automatically validated and typed.                    |
|                                                           | Provider APIs are abstracted for easier switching.           |

```python
# Code Example (as provided in the original README)
import instructor
from pydantic import BaseModel

# Define what you want
class User(BaseModel):
    name: str
    age: int

# Extract it from natural language
client = instructor.from_provider("openai/gpt-4o-mini")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)  # User(name='John', age=25)
```

## Get Started Quickly

Install Instructor with your preferred package manager:

```bash
pip install instructor
```

Or use:

```bash
uv add instructor
poetry add instructor
```

## LLM Provider Agnostic

Use the same code with any LLM provider:

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

## Trusted by the Community

Instructor is a production-ready tool, trusted by:

*   **100,000+ Developers**
*   **3M+ Monthly Downloads**
*   **10K+ GitHub Stars**
*   **1,000+ Community Contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started Now

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

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Choose Instructor?

*   **Superior to Raw JSON Mode:** Instructor offers automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.
*   **More Focused than LangChain/LlamaIndex:** Instructor is streamlined for structured extraction, making it lighter, faster, and easier to debug.
*   **Production-Ready and Battle-Tested:** Trusted by thousands of developers, Instructor handles complex edge cases.

## Contributing

Contributions are welcome! See our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>