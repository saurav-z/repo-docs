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

## ToolFront: Unlock AI's Full Potential with Effortless Data Retrieval

**ToolFront** is a powerful Python library that simplifies data retrieval for AI applications, providing unmatched control, precision, and speed.  Access databases, APIs, and documents with ease, leveraging the power of leading AI models.  Explore the source code on [GitHub](https://github.com/kruskal-labs/toolfront).

**Key Features:**

*   **Seamless Integration:** Connect to various data sources, including databases, APIs (OpenAPI/Swagger compatible), and documents.
*   **AI Model Agnostic:**  Works with a wide range of AI models from providers like OpenAI, Anthropic, Google, xAI, and others.
*   **Structured Output:**  Retrieve data in the format you need, including strings, lists, and structured Pydantic objects, ensuring easy integration with your applications.
*   **Rapid Development:**  Reduce development time with intuitive APIs and streamlined data retrieval workflows.
*   **Flexible and Extensible:** Supports various databases, APIs, and document formats.

---

**Installation**

Install ToolFront using `pip`:

```bash
pip install toolfront
```

For specific database support, install with extras, e.g., `pip install "toolfront[postgres]"`.  See the [documentation](http://docs.toolfront.ai/) for a full list of supported databases.

---

**Examples:**

*   **Text-to-SQL with ChatGPT:**

    ```python
    from toolfront import Database

    db = Database("postgres://user:pass@localhost:5432/mydb", model="openai:gpt-4o")

    context = "We're an e-commerce company. Sales data is in the `cust_orders` table."

    answer = db.ask("What's our best-selling product?", context=context)
    # >>> "Wireless Headphones Pro"
    ```
*   **API Retrieval with Claude:**

    ```python
    from toolfront import API

    api = API("http://localhost:8000/openapi.json", model="anthropic:claude-3-5-sonnet")

    answer: list[int] = api.ask("Get the last 5 order IDs for user_id=42")
    # >>> [1001, 998, 987, 976, 965]
    ```

*   **Document Information Extraction with Gemini:**

    ```python
    from toolfront import Document
    from pydantic import BaseModel, Field

    class CompanyReport(BaseModel):
        company_name: str = Field(..., description="Name of the company")
        revenue: int | float = Field(..., description="Annual revenue in USD")
        is_profitable: bool = Field(..., description="Whether the company is profitable")

    doc = Document("/path/annual_report.pdf", model="google:gemini-pro")

    answer: CompanyReport = doc.ask("Extract the key company information from this report")
    # >>> CompanyReport(company_name="TechCorp Inc.", revenue=2500000, is_profitable=True)
    ```

*   **Snowflake MCP Server (Example Configuration):**

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

---

**Community & Resources**

*   **Documentation:** [docs.toolfront.ai](http://docs.toolfront.ai/)
*   **Discord:** Join our [community server](https://discord.gg/rRyM7zkZTf) for real-time help and discussions
*   **X (Twitter):** Follow us [@toolfront](https://x.com/toolfront) for updates and news
*   **GitHub Issues:** Report bugs or request features on [GitHub Issues](https://github.com/kruskal-labs/toolfront/issues)

---

**License**

This project is licensed under the MIT license.