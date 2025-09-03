<p align="center">
  <a href="https://github.com/kruskal-labs/toolfront">
    <img src="https://raw.githubusercontent.com/kruskal-labs/toolfront/main/img/logo.png" width="150" alt="ToolFront Logo">
  </a>
</p>

<div align="center">

*Simple data retrieval for AI with unmatched control, precision, and speed.*

[![Test Suite](https://github.com/kruskal-labs/toolfront/actions/workflows/test.yml/badge.svg)](https://github.com/kruskal-labs/toolfront/actions/workflows/test.yml)
[![PyPI package](https://img.shields.io/pypi/v/toolfront?color=%2334D058&label=pypi%20package)](https://pypi.org/project/toolfront/)
[![Discord](https://img.shields.io/discord/1323415085011701870?label=Discord&logo=discord&logoColor=white&style=flat-square)](https://discord.gg/rRyM7zkZTf)
[![X](https://img.shields.io/badge/ToolFront-black?style=flat-square&logo=x&logoColor=white)](https://x.com/toolfront)

</div>

---

## ToolFront: Unlock Seamless Data Retrieval for AI Applications

**ToolFront** empowers developers to effortlessly retrieve data from various sources, enabling powerful AI-driven applications. [Explore the ToolFront repository](https://github.com/kruskal-labs/toolfront) to get started.

### Key Features:

*   **Unified Interface:** Access data from databases, APIs, and documents with a consistent and intuitive API.
*   **AI-Powered Data Extraction:** Leverage the power of AI models (OpenAI, Anthropic, Google, and more) for intelligent data retrieval and transformation.
*   **Unmatched Control & Precision:** Fine-tune your data queries and extraction processes for optimal results.
*   **Fast Performance:**  Built for speed and efficiency, allowing for rapid data retrieval.
*   **Extensive Integrations:**  Supports a wide range of databases, APIs (OpenAPI/Swagger compatible), and document formats.
*   **Flexible Deployment:** Easily deploy ToolFront with tools like Snowflake and more.

### Installation

Install ToolFront using pip:

```bash
pip install toolfront
```

For database support, install the necessary extras, for example: `pip install "toolfront[postgres]"`

### Example Usage

ToolFront offers a straightforward approach to data retrieval.  Here are a few examples:

#### 1. Database Interaction (Text-to-SQL with OpenAI)

```python
from toolfront import Database

db = Database("postgres://user:pass@localhost:5432/mydb", model="openai:gpt-4o")

context = "We're an e-commerce company. Sales data is in the `cust_orders` table."

# Returns a string
answer = db.ask("What's our best-selling product?", context=context)
# >>> "Wireless Headphones Pro"
```

#### 2. API Retrieval (with Claude)

```python
from toolfront import API

api = API("http://localhost:8000/openapi.json", model="anthropic:claude-3-5-sonnet")

# Returns a list of integers
answer: list[int] = api.ask("Get the last 5 order IDs for user_id=42")
# >>> [1001, 998, 987, 976, 965]
```

#### 3. Document Information Extraction (with Gemini)

```python
from toolfront import Document
from pydantic import BaseModel, Field

class CompanyReport(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    revenue: int | float = Field(..., description="Annual revenue in USD")
    is_profitable: bool = Field(..., description="Whether the company is profitable")

doc = Document("/path/annual_report.pdf", model="google:gemini-pro")

# Returns a structured Pydantic object
answer: CompanyReport = doc.ask("Extract the key company information from this report")
# >>> CompanyReport(company_name="TechCorp Inc.", revenue=2500000, is_profitable=True)
```

### Deploying with Snowflake (Example)

```json
{
  "mcpServers": {
    "toolfront": {
      "command": "uvx",
      "args": [
        "toolfront[snowflake]", 
        "snowflake://user:pass@account/warehouse/database"
      ]
    }
  }
}
```

### Documentation

For detailed information, examples, and advanced usage, please refer to the official [ToolFront Documentation](http://docs.toolfront.ai/).

### Community & Contributing

*   **Discord:** Join our [Discord server](https://discord.gg/rRyM7zkZTf) for support and discussions.
*   **X (Twitter):** Follow us on [@toolfront](https://x.com/toolfront) for updates.
*   **GitHub Issues:** Report bugs and suggest features on our [GitHub Issues](https://github.com/kruskal-labs/toolfront/issues).

### License

ToolFront is licensed under the [MIT License](https://github.com/kruskal-labs/toolfront/blob/main/LICENSE).