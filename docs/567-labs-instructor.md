# Instructor: Effortlessly Extract Structured Data from LLMs

**Simplify your AI application development by reliably extracting structured data from any Large Language Model with Instructor, eliminating the need for complex JSON parsing and error handling.**

[Go to the original repo](https://github.com/567-labs/instructor)

**Key Features:**

*   **Seamless Integration:** Built on Pydantic for easy validation and type safety.
*   **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, and more).
*   **Automatic Error Handling:** Includes automatic retries for failed validations.
*   **Streaming Support:** Stream partial objects as they're generated.
*   **Nested Object Support:** Extract complex, nested data structures effortlessly.

## What is Instructor?

Instructor streamlines the process of extracting structured data from LLMs. Forget manual JSON parsing, error handling, and complex schema definitions. Instructor simplifies the workflow, enabling you to define your desired data structure with Pydantic models and receive validated, typed data with a clean and intuitive API.

### **The Problem Instructor Solves**

Getting reliable, structured data from LLMs can be challenging. The traditional approach involves:

1.  Writing complex JSON schemas
2.  Manually handling validation errors
3.  Implementing retry mechanisms for failed extractions
4.  Parsing unstructured responses
5.  Adapting to different provider APIs

**Instructor simplifies all of this with a single, unified interface:**

| Without Instructor                                                                      | With Instructor                                                                        |
| :--------------------------------------------------------------------------------------- | :------------------------------------------------------------------------------------- |
| ```python response = openai.chat.completions.create( model="gpt-4", messages=[{"role": "user", "content": "..."}], tools=[ { "type": "function", "function": { "name": "extract_user", "parameters": { "type": "object", "properties": { "name": {"type": "string"}, "age": {"type": "integer"}, }, }, }, } ], ) # Parse response tool_call = response.choices[0].message.tool_calls[0] user_data = json.loads(tool_call.function.arguments) # Validate manually if "name" not in user_data: # Handle error... pass ``` | ```python client = instructor.from_provider("openai/gpt-4") user = client.chat.completions.create( response_model=User, messages=[{"role": "user", "content": "..."}], ) # That's it! user is validated and typed ``` |

## Installation

Get started in seconds:

```bash
pip install instructor
```

Alternatively, use your preferred package manager:

```bash
uv add instructor
poetry add instructor
```

## Provider Support

Instructor offers universal compatibility, allowing you to use the same code with any LLM provider.

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

Instructor automatically retries when validation fails, incorporating the error message for improved results.

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

Receive partial objects as they are generated, enabling real-time data processing.

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

Effortlessly extract complex, nested data structures from your LLM responses.

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

## Trusted by Developers

Instructor is a widely adopted solution, trusted by:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

Used by teams at OpenAI, Google, Microsoft, AWS, and numerous Y Combinator startups.

## Get Started

### Basic Extraction

Quickly extract structured data from any text with Instructor:

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

Instructor's simple API is available in multiple languages:

*   [Python](https://python.useinstructor.com) - The original
*   [TypeScript](https://js.useinstructor.com) - Full TypeScript support
*   [Ruby](https://ruby.useinstructor.com) - Ruby implementation
*   [Go](https://go.useinstructor.com) - Go implementation
*   [Elixir](https://hex.pm/packages/instructor) - Elixir implementation
*   [Rust](https://rust.useinstructor.com) - Rust implementation

### Resources

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Instructor vs. Alternatives

*   **vs Raw JSON mode**: Instructor offers automatic validation, retries, streaming, and nested object support without the need for manual schema writing.
*   **vs LangChain/LlamaIndex**: Instructor focuses solely on structured extraction, resulting in a lighter, faster, and more easily debugged solution.
*   **vs Custom solutions**: Benefit from a battle-tested library, proven by thousands of developers, handling edge cases effectively.

## Contributing

We welcome contributions!  Explore our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>