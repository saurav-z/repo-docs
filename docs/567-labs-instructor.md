# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your LLM interactions and get reliable, validated JSON outputs effortlessly with Instructor, built on Pydantic.**

[Go to the original repository](https://github.com/567-labs/instructor)

Instructor makes extracting structured data from Language Learning Models (LLMs) easy. With validation, type safety, and robust error handling, you can build more reliable and maintainable AI applications.

Key Features:

*   ✅ **Pydantic Integration:** Seamlessly integrates with Pydantic for data validation and type safety, offering a familiar and powerful development experience.
*   ✅ **Automatic Retries:**  Handles LLM failures automatically, including validation errors, ensuring more robust and reliable extractions.
*   ✅ **Streaming Support:** Stream partial objects as they're generated, enabling real-time data processing.
*   ✅ **Nested Object Support:** Effortlessly extracts complex, nested data structures from LLMs.
*   ✅ **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama) using a unified API.
*   ✅ **Production-Ready:** Used by over 100,000 developers and companies, processing millions of requests per month.

## Why Choose Instructor?

Instructor simplifies the process of extracting structured data compared to manual approaches or other frameworks.  It eliminates the need for:

*   Complex JSON schemas
*   Manual validation error handling
*   Tedious retry logic
*   Parsing unstructured responses
*   Provider-specific API implementations

**Instructor offers a streamlined solution:**

| Without Instructor                                                                                                                                                                                     | With Instructor                                                                                                  |
| :----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | :--------------------------------------------------------------------------------------------------------------- |
| `response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[{"type": "function", "function": {"name": "extract_user", "parameters": {...}}}], )`<br/>...manual parsing, validation, and error handling... | `client = instructor.from_provider("openai/gpt-4")`<br/><br/>`user = client.chat.completions.create(response_model=User, messages=[{"role": "user", "content": "..."}],)` |

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or with your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Example: Basic Data Extraction

Easily extract structured data:

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

print(product)  # Product(name='iPhone 15 Pro', price=999.0, in_stock=True)
```

## Advanced Features

### Automatic Retries

Instructor automatically retries failed extractions.

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

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

##  Works with All Major LLM Providers

Use the same Instructor code with any LLM provider.

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

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

##  Used in Production By

Instructor is a trusted solution, adopted by over 100,000 developers, including teams at OpenAI, Google, Microsoft, AWS, and numerous startups.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Get Started

### Python Example

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

Instructor is also available in multiple languages:

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

## Alternatives Comparison

**vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.

**vs LangChain/LlamaIndex:** Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.

**vs Custom solutions:** Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contribute

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>