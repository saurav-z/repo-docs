# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your AI workflow and get reliable, structured JSON from any Language Learning Model (LLM) with Instructor, built on Pydantic for type safety and robust validation.**

[Link to Original Repo: https://github.com/567-labs/instructor](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features of Instructor

*   **Seamless Integration:** Effortlessly extract structured data without complex JSON schema writing or manual parsing.
*   **Pydantic Powered:** Leverages Pydantic for type safety, validation, and IDE support.
*   **Automatic Error Handling:** Built-in automatic retries for failed extractions.
*   **Multi-Provider Support:** Works with all major LLM providers, including OpenAI, Anthropic, Google, and local models.
*   **Streaming Support:** Get partial objects as they're generated, improving responsiveness.
*   **Nested Object Extraction:** Easily handle complex, nested data structures.
*   **Production-Ready:** Designed for production environments, with automatic retries and robust error handling.

## The Problem Instructor Solves

Extracting structured data from LLMs can be a challenging process, involving:

1.  Writing complex JSON schemas
2.  Handling validation errors
3.  Retrying failed extractions
4.  Parsing unstructured responses
5.  Dealing with different provider APIs

**Instructor streamlines this entire process with a simple interface.**

| **Without Instructor**                                   | **With Instructor**                                       |
| :------------------------------------------------------- | :-------------------------------------------------------- |
| *Example of complex, manual extraction process...*   | *Example of clean, concise Instructor implementation...* |

## Getting Started

### Installation

Install Instructor quickly using pip:

```bash
pip install instructor
```

Or your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Usage

Extract structured data with ease:

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

## Production-Ready Capabilities

Instructor is designed for production environments, offering features like:

### Automatic Retries

Automatically retry when validation fails, increasing reliability:

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

## Multi-Language Support

Instructor's simple API is available in several languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

## Why Choose Instructor?

*   **Simplified Workflow:** Instructor simplifies structured data extraction, eliminating manual parsing and schema creation.
*   **Robust Validation:** Pydantic integration ensures data integrity with built-in validation.
*   **Increased Efficiency:** Automatic retries and streaming support enhance your LLM interactions.
*   **Provider Agnostic:** Works seamlessly with various LLM providers, offering flexibility.

**Instructor vs. Alternatives:**

*   **vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.
*   **vs LangChain/LlamaIndex**: Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.
*   **vs Custom solutions**: Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Used in Production By

Instructor is trusted by over 100,000 developers and companies building AI applications.

-   **3M+ monthly downloads**
-   **10K+ GitHub stars**
-   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>