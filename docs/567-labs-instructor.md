# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your interactions with Large Language Models by effortlessly extracting structured data with Instructor, built for reliability, type safety, and ease of use.**  This library makes it easy to get reliable JSON from any LLM.

[Visit the original repository](https://github.com/567-labs/instructor) for more details.

[![PyPI](https://img.shields.io/pypi/v/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![Downloads](https://img.shields.io/pypi/dm/instructor?style=flat-square)](https://pypi.org/project/instructor/)
[![GitHub Stars](https://img.shields.io/github/stars/instructor-ai/instructor?style=flat-square)](https://github.com/instructor-ai/instructor)
[![Discord](https://img.shields.io/discord/1192334452110659664?style=flat-square)](https://discord.gg/bD9YE9JArw)
[![Twitter](https://img.shields.io/twitter/follow/jxnlco?style=flat-square)](https://twitter.com/jxnlco)

## Key Features

*   **Simplified Data Extraction:** Easily define models using Pydantic and get structured data without complex JSON parsing or error handling.
*   **Automatic Error Handling:** Includes automatic retries for failed extractions, ensuring robust results.
*   **Provider Agnostic:** Works seamlessly with major LLM providers including OpenAI, Anthropic, Google, and local models like Ollama.
*   **Production-Ready Features:** Includes features like automatic retries, streaming support, and nested object support.
*   **Type Safety & IDE Support:** Built with Pydantic, providing type safety and full IDE support for enhanced development.

## Why Choose Instructor?

Instructor streamlines the process of extracting structured data from LLMs, handling the complexities of:

*   Complex JSON schema creation
*   Validation errors
*   Extraction failures
*   Unstructured response parsing
*   API differences between providers

| <b>Without Instructor</b>                                                                                                                                 | <b>With Instructor</b>                                                                                                                                     |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------- |
| ```python <br> response = openai.chat.completions.create( <br> &nbsp;&nbsp; model="gpt-4", <br> &nbsp;&nbsp; messages=[{"role": "user", "content": "..."}], <br> &nbsp;&nbsp; tools=[ <br> &nbsp;&nbsp;&nbsp;&nbsp; { <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "type": "function", <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "function": { <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "name": "extract_user", <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "parameters": { <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "type": "object", <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "properties": { <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "name": {"type": "string"}, <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; "age": {"type": "integer"}, <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; }, <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; }, <br> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; }, <br> &nbsp;&nbsp; ], <br> ) <br> <br> # Parse response <br> tool_call = response.choices[0].message.tool_calls[0] <br> user_data = json.loads(tool_call.function.arguments) <br> <br> # Validate manually <br> if "name" not in user_data: <br> &nbsp;&nbsp; # Handle error... <br> &nbsp;&nbsp; pass <br> ``` | ```python <br> client = instructor.from_provider("openai/gpt-4") <br> <br> user = client.chat.completions.create( <br> &nbsp;&nbsp; response_model=User, <br> &nbsp;&nbsp; messages=[{"role": "user", "content": "..."}], <br> ) <br> <br> # That's it! user is validated and typed <br> ``` |

## Installation

Install Instructor in seconds using pip:

```bash
pip install instructor
```

Or with your package manager:

```bash
uv add instructor
poetry add instructor
```

## Supported Providers

Instructor offers a unified API across various LLM providers, allowing you to easily switch between them:

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

## Dive Deeper

### Basic Extraction

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

### Production-Ready Features

*   **Automatic Retries:** Automatically retries failed validations.
*   **Streaming Support:** Stream partial objects as they're generated.
*   **Nested Objects:** Extract complex, nested data structures effortlessly.

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

## Used in Production By

Instructor is trusted by over 100,000 developers and companies to build AI applications, including:

*   3M+ monthly downloads
*   10K+ GitHub stars
*   1000+ community contributors

Companies using Instructor include teams at OpenAI, Google, Microsoft, AWS, and many YC startups.

## Get Started

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Why Instructor Over Alternatives?

*   **vs Raw JSON mode:** Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.
*   **vs LangChain/LlamaIndex:** Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.
*   **vs Custom solutions:** Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>