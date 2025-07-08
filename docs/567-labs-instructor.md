# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify LLM interactions and get reliable, validated JSON outputs with Instructor, a powerful Python library built on Pydantic.**  [Visit the original repository](https://github.com/567-labs/instructor)

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Easy Integration:** Seamlessly integrate with leading LLM providers, including OpenAI, Anthropic, Google, and local models like Ollama.
*   **Pydantic Validation:** Leverage the power of Pydantic for robust data validation, type safety, and IDE support.
*   **Automatic Retries:** Handles LLM response errors with automatic retries, ensuring reliable extractions.
*   **Streaming Support:** Receive and process partial objects in real-time with streaming capabilities.
*   **Nested Object Extraction:** Effortlessly extract complex, nested data structures.
*   **Multi-Language Support:** Available in Python, TypeScript, Ruby, Go, Elixir, and Rust, providing flexibility for diverse projects.

## Why Choose Instructor?

Instructor solves the common pain points of extracting structured data from LLMs:

*   **Simplified Workflow:** Avoid complex JSON schema creation and manual parsing.
*   **Reliable Data:**  Benefit from automatic validation and error handling, ensuring data quality.
*   **Production-Ready:** Built-in features like automatic retries and streaming make Instructor suitable for production environments.

**Instructor vs. Alternatives:**

*   **Raw JSON Mode:** Instructor automates validation, retries, streaming, and nested object support – without manual schema creation.
*   **LangChain/LlamaIndex:** Instructor offers a focused, lightweight, and easier-to-debug solution specifically for structured extraction.
*   **Custom Solutions:** Instructor offers a battle-tested solution, addressing edge cases and complexities that custom solutions often miss.

## Getting Started

### Installation

Install Instructor in seconds:

```bash
pip install instructor
```

Or using your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

### Basic Usage

Define your data model and extract structured information:

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

### Production-Ready Features

*   **Automatic Retries:** Instructor automatically retries failed extractions.

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

*   **Streaming Support:** Stream partial objects as they're generated.

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

*   **Nested Objects:** Extract complex, nested data structures.

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

Trusted by over 100,000 developers and companies building AI applications:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>
```
Key improvements and SEO considerations:

*   **Clear, Concise Title:**  "Instructor: Effortlessly Extract Structured Data from LLMs" is more descriptive and keyword-rich.
*   **One-Sentence Hook:** The opening sentence immediately highlights Instructor's core value proposition.
*   **Keyword Optimization:**  Includes keywords like "structured data," "LLMs," "JSON," "Pydantic," and specific LLM providers to improve search visibility.
*   **Bulleted Key Features:**  Easy-to-scan format for users.
*   **Comparison Table Enhanced:** Provides a clearer contrast between using Instructor and manual methods.
*   **Concise Examples:** Kept the examples clean and focused.
*   **Clearer "Why Use" Section:**  More persuasive and value-driven.
*   **Links to Resources:**  Added links to the original repo.
*   **Formatting:**  Improved use of Markdown for better readability and SEO.
*   **Headings:** Uses headings for better organization and readability.
*   **Call to Action:** Encourages users to "Get Started".
*   **Summarized Key Information:** Condenses the information while retaining the important aspects.
*   **Emphasis on Benefits:** Highlights the advantages of using Instructor.
*   **Removed Redundancy**: Streamlined content for brevity without sacrificing clarity.