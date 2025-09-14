# Instructor: Get Reliable JSON from LLMs - Effortlessly!

**Simplify LLM integrations with Instructor, the Python library that empowers you to extract structured data reliably and efficiently from any language model.** ([Original Repo](https://github.com/567-labs/instructor))

Instructor simplifies the process of extracting structured data from language models. Built on Pydantic for robust validation, type safety, and excellent IDE support, Instructor eliminates the complexities of JSON parsing, error handling, and retries, allowing you to focus on building your AI applications.

**Key Features:**

*   ✅ **Simplified API:** Define your desired output model (using Pydantic), and let Instructor handle the rest.
*   ✅ **Automatic Validation:** Ensures data integrity with built-in validation and automatic retries on failure.
*   ✅ **Provider Agnostic:** Works seamlessly with OpenAI, Anthropic, Google, Ollama, and more.
*   ✅ **Streaming Support:** Receive partial objects in real-time as the LLM generates them.
*   ✅ **Nested Object Support:** Handles complex, nested data structures effortlessly.

```python
import instructor
from pydantic import BaseModel

class User(BaseModel):
    name: str
    age: int

client = instructor.from_provider("openai/gpt-4o-mini")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "John is 25 years old"}],
)

print(user)  # User(name='John', age=25)
```

## Why Choose Instructor?

Instructor significantly simplifies the process of extracting structured data from LLMs, offering a streamlined approach compared to traditional methods:

*   **No More Complex JSON Schemas:** Avoid the hassle of manually defining and managing intricate JSON schemas.
*   **Automatic Error Handling:** Eliminate the need to write custom error handling and retry mechanisms.
*   **Seamless Integration:** Easily integrate with various LLM providers using a consistent API.

## Install Instructor

Get started in seconds:

```bash
pip install instructor
```

## Supported LLM Providers

Instructor provides broad support for popular LLM providers:

*   OpenAI
*   Anthropic
*   Google
*   Ollama (local)

```python
# Example Usage:
client = instructor.from_provider("openai/gpt-4o")
user = client.chat.completions.create(
    response_model=User,
    messages=[{"role": "user", "content": "..."}],
)
```

## Production-Ready Features

Instructor offers features to improve reliability:

*   **Automatic Retries**: Automatically retries failed extractions.
*   **Streaming Support**: Stream partial objects as they're generated.
*   **Nested Objects**: Easily handle complex, nested data structures.

## Trusted by Developers

Instructor is a community-driven project used by over 100,000 developers and companies, including those at OpenAI, Google, Microsoft, and more:

*   **3M+ monthly downloads**
*   **10K+ GitHub stars**
*   **1000+ community contributors**

## Get Started

*   [Documentation](https://python.useinstructor.com) - Comprehensive guides
*   [Examples](https://python.useinstructor.com/examples/) - Copy-paste recipes
*   [Blog](https://python.useinstructor.com/blog/) - Tutorials and best practices
*   [Discord](https://discord.gg/bD9YE9JArw) - Get help from the community

## Comparison

Instructor provides significant advantages over alternative solutions:

*   **vs Raw JSON mode**: Instructor provides automatic validation, retries, streaming, and nested object support. No manual schema writing.
*   **vs LangChain/LlamaIndex**: Instructor is focused on one thing - structured extraction. It's lighter, faster, and easier to debug.
*   **vs Custom solutions**: Battle-tested by thousands of developers. Handles edge cases you haven't thought of yet.

## Contribute

We welcome contributions! Check out our [good first issues](https://github.com/instructor-ai/instructor/labels/good%20first%20issue) to get started.

## License

MIT License - see [LICENSE](https://github.com/instructor-ai/instructor/blob/main/LICENSE) for details.

---

<p align="center">
Built by the Instructor community. Special thanks to <a href="https://twitter.com/jxnlco">Jason Liu</a> and all <a href="https://github.com/instructor-ai/instructor/graphs/contributors">contributors</a>.
</p>