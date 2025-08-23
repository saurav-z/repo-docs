<div align="center">
  <img alt="FastAPI Best Architecture Logo" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">
</div>

# FastAPI Best Architecture: Build Robust Backend Applications

This project offers a comprehensive, enterprise-level architecture solution for building powerful and scalable backends with FastAPI. For more details, please see the [original repository](https://github.com/fastapi-practices/fastapi_best_architecture).

[![GitHub License](https://img.shields.io/github/license/fastapi-practices/fastapi_best_architecture)](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
![MySQL](https://img.shields.io/badge/MySQL-8.0%2B-%2300758f)
![PostgreSQL](https://img.shields.io/badge/PostgreSQL-16.0%2B-%23336791)
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-%23778877)
[![Pydantic v2](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/pydantic/pydantic/main/docs/badge/v2.json)](https://pydantic.dev)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
![Docker](https://img.shields.io/badge/Docker-%232496ED?logo=docker&logoColor=white)
[![Discord](https://img.shields.io/badge/Discord-%235865F2.svg?logo=discord&logoColor=white)](https://discord.com/invite/yNN3wTbVAC)
![Discord](https://img.shields.io/discord/1185035164577972344)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/fastapi-practices/fastapi_best_architecture)

## Key Features

*   **Pseudo 3-Tier Architecture:** Leverages a well-defined structure for clear separation of concerns, inspired by 3-tier architecture principles:
    *   **API (View):**  Handles incoming requests and returns responses.
    *   **Schema (Data Transfer):** Defines data structures for input and output using Pydantic.
    *   **Service (Business Logic):** Implements business rules and orchestrates operations.
    *   **CRUD (Data Access):**  Manages database interactions.
    *   **Model:** Represents the data structure.
*   **Built on Modern Technologies:** Uses FastAPI, Pydantic, SQLAlchemy, and other cutting-edge tools for performance, validation, and data management.
*   **Database Support:**  Offers support for MySQL and PostgreSQL.
*   **Code Quality:** Employs tools like Ruff for code linting and formatting.
*   **Containerization:**  Includes Docker support for easy deployment.

## Architecture Overview

This project implements a "pseudo 3-tier" architecture, with components mapped to the following:

| Workflow       | Java           | FastAPI Best Architecture |
|----------------|----------------|---------------------------|
| View           | Controller     | API                       |
| Data Transfer  | DTO            | Schema                    |
| Business Logic | Service + Impl | Service                   |
| Data Access    | DAO / Mapper   | CRUD                      |
| Model          | Model / Entity | Model                     |

## Documentation & Resources

*   **Official Documentation:**  [https://fastapi-practices.github.io/fastapi_best_architecture_docs/](https://fastapi-practices.github.io/fastapi_best_architecture_docs/)

## Get Involved

*   **Discord:** [https://wu-clan.github.io/homepage/](https://wu-clan.github.io/homepage/)
*   **Contributors:**  See the [contributors graph](https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors)

## Special Thanks

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)
*   ...

## Support the Project

If you find this project helpful, consider supporting it with a coffee donation: [:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

## License

This project is licensed under the [MIT License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE).

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)