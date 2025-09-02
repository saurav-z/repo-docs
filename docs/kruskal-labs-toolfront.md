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

## ToolFront: Effortlessly Retrieve Data for AI Applications

ToolFront empowers you to seamlessly access and integrate data from databases, APIs, and documents directly into your AI workflows.  [Explore the ToolFront repository on GitHub](https://github.com/kruskal-labs/toolfront).

**Key Features:**

*   **Versatile Data Sources:** Connect to databases (Postgres, Snowflake, and more), APIs (OpenAPI-compatible), and documents (PDFs, etc.).
*   **AI Model Agnostic:**  Integrates with leading AI model providers like OpenAI, Anthropic, Google, and xAI.
*   **Structured Output:**  Receive data in a structured format (e.g., JSON, Pydantic models) for easy use in your AI applications.
*   **Unmatched Control & Precision:**  Fine-tune data retrieval with context and specific instructions.
*   **Rapid Deployment:**  Simple installation with `pip` and straightforward integration into your projects.

---

## Quickstart: Installation

Install ToolFront using pip:

```bash
pip install toolfront
```

For specific database support, install the relevant extras (e.g., `pip install "toolfront[postgres]"`).  See the [documentation](http://docs.toolfront.ai/) for a full list.

---

## Examples

### Database Integration (Text-to-SQL)

```python
from toolfront import Database

db = Database("postgres://user:pass@localhost:5432/mydb", model="openai:gpt-4o")

context = "We're an e-commerce company. Sales data is in the `cust_orders` table."

# Returns a string
answer = db.ask("What's our best-selling product?", context=context)
# >>> "Wireless Headphones Pro"
```

### API Interaction

```python
from toolfront import API

api = API("http://localhost:8000/openapi.json", model="anthropic:claude-3-5-sonnet")

# Returns a list of integers
answer: list[int] = api.ask("Get the last 5 order IDs for user_id=42")
# >>> [1001, 998, 987, 976, 965]
```

### Document Processing

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

---

## Community & Resources

*   **Documentation:** [docs.toolfront.ai](http://docs.toolfront.ai/)
*   **Discord:** Join our [community server](https://discord.gg/rRyM7zkZTf)
*   **X (Twitter):** Follow us [@toolfront](https://x.com/toolfront)
*   **GitHub Issues:** Report bugs or request features on [GitHub Issues](https://github.com/kruskal-labs/toolfront/issues)

---

## License

This project is licensed under the MIT License.