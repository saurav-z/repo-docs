<div align="center">
  <img alt="FastAPI Best Architecture Logo" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">
</div>

# FastAPI Best Architecture: Build Robust Backend Systems with Python

**Looking for a production-ready, enterprise-level backend architecture solution with FastAPI?** This project provides a robust, well-structured foundation for building scalable and maintainable APIs using Python and FastAPI.  [View the original repository on GitHub](https://github.com/fastapi-practices/fastapi_best_architecture).

[![GitHub](https://img.shields.io/github/license/fastapi-practices/fastapi_best_architecture)](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue)](https://www.python.org/downloads/)
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

*   **Pseudo 3-Tier Architecture:**  A clear, well-defined structure for organizing your application logic, inspired by 3-tier architecture principles.
*   **FastAPI Integration:** Leverages the power and speed of FastAPI for building high-performance APIs.
*   **Data Modeling:** Utilizes SQLAlchemy and Pydantic for robust data modeling and validation.
*   **Database Support:** Supports popular databases including MySQL and PostgreSQL.
*   **Modern Tooling:** Integrates with Ruff and `uv` for linting, formatting, and dependency management.
*   **Docker Support:** Provides Docker configuration for easy deployment and containerization.

## Architecture Overview

This project adopts a "pseudo 3-tier" architecture to provide a clear separation of concerns:

| Workflow        |  Component              | Description                                               |
|-----------------|-------------------------|-----------------------------------------------------------|
| View            | API                     | Handles incoming requests and returns responses.         |
| Data Transfer   | Schema                  | Defines the structure and validation of data payloads.   |
| Business Logic  | Service                 | Contains the core business logic and operations.         |
| Data Access     | CRUD                    | Manages database interactions (Create, Read, Update, Delete). |
| Model           | Model                   | Represents the data structure in the database.          |

## Documentation

Explore the complete project documentation for detailed information and usage instructions:  [Official Documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/)

##  Contributing

We welcome contributions!  See the project's contribution guidelines for more information.

<a href="https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=fastapi-practices/fastapi_best_architecture"/>
</a>

## Special Thanks

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)
*   ...

## Get Involved

*   [Discord](https://wu-clan.github.io/homepage/)

## Support the Project

If you find this project helpful, consider supporting us with a coffee! [:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

## License

This project is licensed under the [MIT License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE).

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)