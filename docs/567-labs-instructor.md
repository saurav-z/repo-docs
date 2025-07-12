# Instructor: Effortless Structured Data Extraction from LLMs

**Simplify your AI development and get reliable, validated JSON from any Large Language Model (LLM) with Instructor, built on Pydantic for type safety and seamless integration.** ([See the original repo](https://github.com/567-labs/instructor))

Instructor streamlines the process of extracting structured data from LLMs, eliminating the need for complex JSON parsing and manual error handling.

## Key Features

*   ✅ **Pydantic Integration:** Leverage Pydantic for type safety, validation, and IDE support.
*   ✅ **Automatic Error Handling:** Includes automatic retries for failed extractions.
*   ✅ **Provider Agnostic:** Works with all major LLM providers (OpenAI, Anthropic, Google, Ollama, and more) using a unified API.
*   ✅ **Streaming Support:** Receive partial objects as they are generated for improved user experience.
*   ✅ **Nested Objects:** Easily extract complex, nested data structures.

## Get Started in Seconds

Install Instructor using your preferred package manager:

```bash
pip install instructor
```
```bash
# or with other package managers:
uv add instructor
poetry add instructor
```

## Core Functionality

### Extract Structured Data

Define a Pydantic model and let Instructor handle the extraction:

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

*   **Automatic Retries:** Instructor automatically retries when validation fails, incorporating error messages.
*   **Streaming Support:** Retrieve partial objects as they are generated.
*   **Nested Objects:** Supports extraction of complex and nested data structures.

### Provider Compatibility

Use the same code across various LLM providers:

```python
client = instructor.from_provider("openai/gpt-4o")
client = instructor.from_provider("anthropic/claude-3-5-sonnet")
client = instructor.from_provider("google/gemini-pro")
client = instructor.from_provider("ollama/llama3.2")

# All use the same API!
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

##  Why Use Instructor?

**Instructor offers significant advantages over manual JSON parsing, LangChain, and other alternatives.**

*   **Simplified Development:** Reduces boilerplate code and simplifies your workflow.
*   **Improved Reliability:** Ensures data validation and handles common extraction issues.
*   **Enhanced Efficiency:** Speeds up development time and improves the performance of your AI applications.

## Join the Community

Instructor is trusted by 100,000+ developers and companies.

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**
*   **Discord:** Join the community [here](https://discord.gg/bD9YE9JArw)

## Resources

*   **Documentation:** Comprehensive guides and examples on [python.useinstructor.com](https://python.useinstructor.com)
*   **Examples:** Explore copy-paste recipes at [python.useinstructor.com/examples/](https://python.useinstructor.com/examples/)
*   **Blog:** Tutorials and best practices on [python.useinstructor.com/blog/](https://python.useinstructor.com/blog/)
*   **Discord:** Get help from the community [here](https://discord.gg/bD9YE9JArw)
*   **Other Languages:**
    *   [TypeScript](https://js.useinstructor.com)
    *   [Ruby](https://ruby.useinstructor.com)
    *   [Go](https://go.useinstructor.com)
    *   [Elixir](https://hex.pm/packages/instructor)
    *   [Rust](https://rust.useinstructor.com)

## Contributing

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>