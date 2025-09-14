<div align="center">
  <img alt="FastAPI Best Architecture Logo" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">

  # FastAPI Best Architecture: Build Robust & Scalable Backends

</div>

This project provides a comprehensive and opinionated architecture for building enterprise-level backends with FastAPI, designed for clarity, maintainability, and scalability.  Explore the [original repository on GitHub](https://github.com/fastapi-practices/fastapi_best_architecture) for more details.

**Key Features:**

*   **Pseudo 3-Tier Architecture:** Implements a refined architectural pattern inspired by 3-tier design for logical separation of concerns, including:
    *   **API (View):** Handles incoming requests and responses.
    *   **Schema (Data Transfer):** Defines data structures using Pydantic.
    *   **Service (Business Logic):** Encapsulates business rules and operations.
    *   **CRUD (Data Access):** Manages database interactions.
    *   **Model (Data Representation):** Defines database entities.
*   **Modern Tech Stack:** Leverages cutting-edge technologies for optimal performance and developer experience:
    *   Python 3.10+
    *   FastAPI
    *   SQLAlchemy (with support for MySQL and PostgreSQL)
    *   Pydantic v2
    *   Ruff (linter/formatter)
    *   uv (package manager)
    *   Docker
*   **Clear Structure:**  Designed to provide a clear and organized project structure, making it easier to understand, maintain, and extend.
*   **Comprehensive Documentation:** Detailed information available in the [official documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/).

**Architecture Breakdown (Comparison):**

| Workflow         | Java           | FastAPI Best Architecture |
|------------------|----------------|---------------------------|
| View             | Controller     | API                       |
| Data Transfer    | DTO            | Schema                    |
| Business Logic   | Service + impl | Service                   |
| Data Access      | DAO / Mapper   | CRUD                      |
| Model            | Model / Entity | Model                     |

**Getting Help and Contributing:**

*   **Discord:** [Join the community](https://discord.com/invite/yNN3wTbVAC)
*   **Contributors:**  View the project's contributors [here](https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors).

**Special Thanks:**

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)
*   ...

**Support Us:**

*   [Sponsor with Coffee](https://wu-clan.github.io/sponsor/)

**License:**

This project is licensed under the [MIT License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE).

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)