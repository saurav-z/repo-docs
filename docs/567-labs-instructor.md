# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your LLM integrations and reliably obtain structured JSON outputs with Instructor, built on Pydantic and offering validation, retries, and streaming support.** ([Original Repo](https://github.com/567-labs/instructor))

## Key Features

*   **Simplified Extraction:**  Define your desired data structure with Pydantic models and effortlessly extract it from any LLM response.
*   **Automatic Validation:**  Ensure data integrity with built-in validation, catching errors early and improving reliability.
*   **Intelligent Retries:**  Automatically retry extraction attempts when validation fails, enhancing robustness.
*   **Streaming Support:**  Receive and process partial objects as they are generated for a more responsive user experience.
*   **Nested Object Handling:**  Seamlessly extract complex, nested data structures without extra configuration.
*   **Provider Agnostic:** Works with major LLM providers (OpenAI, Anthropic, Google, Ollama, and more) using the same API.

## Why Instructor?

Instructor simplifies the process of working with LLMs, offering:

*   **No more manual JSON parsing:** Eliminate error-prone and time-consuming parsing logic.
*   **Reduced Error Handling:** Handle validation errors automatically with retries.
*   **Simplified Integration:**  Use a unified interface for various LLM providers.
*   **Production-Ready:**  Leverage features like automatic retries and streaming for reliable performance.

## Install in Seconds

```bash
pip install instructor
```

or use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Usage Examples

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

### Production-Ready Features

*   **Automatic Retries:**
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
    messages=[{"role": "user", "content": "John is -25 years old"}],
    max_retries=3,
)
```

*   **Streaming Support:**

```python
from instructor import Partial

for partial_user in client.chat.completions.create(
    response_model=Partial[User],
    messages=[{"role": "user", "content": "..."}],
    stream=True,
):
    print(partial_user)
```

*   **Nested Objects:**

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

## Used in Production

Trusted by over 100,000 developers and companies, including teams at OpenAI, Google, and Microsoft.  

## Learn More

*   [Documentation](https://python.useinstructor.com)
*   [Examples](https://python.useinstructor.com/examples/)
*   [Blog](https://python.useinstructor.com/blog/)
*   [Discord](https://discord.gg/bD9YE9JArw)

## Contributing

Contributions are welcome! Check out the [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue).

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.