# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions by reliably extracting JSON and other structured data with Instructor, built on Pydantic for robust validation and type safety.**

[Visit the original repository on GitHub](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Structured Output:** Easily define data models and extract structured data without complex JSON parsing or schema creation.
*   **Pydantic Integration:** Leverages Pydantic for robust validation, type safety, and seamless IDE support.
*   **Automatic Retries:** Handles validation errors automatically, retrying failed extractions for increased reliability.
*   **Streaming Support:** Stream partial objects as they're generated, enabling real-time data extraction.
*   **Nested Object Extraction:**  Effortlessly extract complex, nested data structures from LLM responses.
*   **Cross-Provider Compatibility:** Works seamlessly with all major LLM providers, including OpenAI, Anthropic, Google, and local models like Ollama.

## The Problem Instructor Solves

Getting structured data from LLMs is notoriously difficult. Existing solutions often involve:

*   Writing and maintaining intricate JSON schemas.
*   Manually handling validation errors and retries.
*   Parsing unstructured responses.
*   Adapting to various provider APIs.

**Instructor streamlines this process with a simple, intuitive interface, drastically reducing complexity and improving reliability.**

## Get Started

### Install

```bash
pip install instructor
```

Or with your package manager:
```bash
uv add instructor
poetry add instructor
```

### Basic Usage

Define your desired data structure and extract it from natural language:

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

### Automatic Retries and Validation

Instructor automatically retries when validation fails, using the error message to improve performance:

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

Stream partial objects in real-time:

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

## Production-Ready Features

Instructor offers production-ready features, including:

*   **Automatic Retries:**  Ensures resilience against LLM response variations.
*   **Streaming Support:** Provides real-time data extraction for improved user experience.
*   **Nested Object Extraction:**  Handles complex data structures with ease.

## Why Instructor?

*   **vs Raw JSON mode:**  Instructor provides automatic validation, retries, streaming, and nested object support.  Eliminates manual schema writing.
*   **vs LangChain/LlamaIndex:**  Instructor is laser-focused on structured extraction, making it lighter, faster, and easier to debug.
*   **vs Custom solutions:**  Benefit from a battle-tested solution used by thousands of developers, covering edge cases and providing a robust foundation.

## Used in Production By

Trusted by over 100,000 developers and companies building AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Additional Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Contributing

We welcome contributions!  See our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>