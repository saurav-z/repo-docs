# Instructor: Effortlessly Get Structured Data from LLMs

**Instructor is a Python library that simplifies extracting structured data from Large Language Models (LLMs), providing reliability, type safety, and ease of use.**

[View the original repo on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Define Python models using Pydantic and let Instructor handle the rest.
*   **Automatic Validation:** Ensures data integrity and type safety, preventing errors.
*   **Intelligent Retries:** Automatically retries failed extractions, improving reliability.
*   **Streaming Support:** Receive partial objects as they're generated, enhancing user experience.
*   **Nested Object Support:** Seamlessly handle complex, nested data structures.
*   **Provider Agnostic:** Works with a wide range of LLM providers, including OpenAI, Anthropic, Google, and local models (Ollama).
*   **Production-Ready:** Built for scale with features like automatic retries and nested object support.
*   **Multi-Language Support**: Also available in [TypeScript](https://js.useinstructor.com), [Ruby](https://ruby.useinstructor.com), [Go](https://go.useinstructor.com), [Elixir](https://hex.pm/packages/instructor), and [Rust](https://rust.useinstructor.com).

## Why Choose Instructor?

Instructor streamlines the process of getting structured data from LLMs by eliminating the need for manual JSON parsing, error handling, and complex schema definitions. It provides a simpler, more reliable, and efficient solution.

| **Without Instructor** | **With Instructor** |
|-----------------------|----------------------|
|

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

|

```python
client = instructor.from_provider("openai/gpt-4")

user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)

# That's it! user is validated and typed
```

|

## Installation

Install Instructor in seconds:

```bash
pip install instructor
```

Or with your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Get Started: Basic Extraction

Effortlessly extract structured data:

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

Stream partial objects:

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

Extract complex, nested data:

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

## Used in Production By

Trusted by over 100,000 developers and companies:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Learn More

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Instructor vs. Alternatives

*   **vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support, eliminating manual schema writing.
*   **vs LangChain/LlamaIndex**: Instructor is focused specifically on structured extraction, offering a lighter, faster, and easier-to-debug solution.
*   **vs Custom solutions**: Instructor provides a battle-tested solution for handling complex scenarios and edge cases.

## Contributing

We welcome contributions! Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>
```
Key improvements and SEO considerations:

*   **Headline Optimization:**  The primary headline includes the target keyword ("structured data from LLMs") to help search engines understand the core functionality.
*   **Concise Hook:** The one-sentence hook effectively captures the core value proposition.
*   **Keyword Integration:** The text uses relevant keywords such as "LLMs," "Pydantic," "validation," "data extraction," and provider names.
*   **Clear Headings and Structure:** The use of clear headings and bullet points improves readability and SEO.  Search engines favor structured content.
*   **Comparison Table:** The "Without Instructor" vs. "With Instructor" table makes the benefit immediately obvious.
*   **Provider Mentions:** Explicitly lists the LLM providers the library supports, and includes the key benefit that it is *provider agnostic*.
*   **Call to Action:** Clear "Installation" and "Get Started" sections encourage immediate use.
*   **Use Case Examples:** Included examples showcase the main features of Instructor.
*   **Emphasis on Benefits:** The text highlights the benefits of using Instructor over alternatives.
*   **Internal Linking:** The inclusion of links to the documentation, examples, and Discord community boosts SEO.
*   **Author and Contributor Credit:** Provides a clear signal that this project is maintained and actively developed.
*   **Multi-Language Support**: Highlights the different languages available and links to their specific documentations.