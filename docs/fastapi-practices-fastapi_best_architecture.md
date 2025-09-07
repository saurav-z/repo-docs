<div align="center">
  <img alt="FastAPI Best Architecture Logo" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">
</div>

# FastAPI Best Architecture: Build Robust Backend Solutions

**Looking for a production-ready architecture for your FastAPI projects?**  This project provides a comprehensive, enterprise-level backend architecture solution built with FastAPI, designed for scalability, maintainability, and ease of development.  Check out the [original repository](https://github.com/fastapi-practices/fastapi_best_architecture) for the full code and more details.

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

*   **Robust Architecture:**  Implements a "pseudo 3-tier" architecture, providing a clear separation of concerns for API, data transfer, business logic, data access, and model layers.
*   **Technology Stack:** Leverages industry-standard technologies including:
    *   FastAPI
    *   Pydantic v2
    *   SQLAlchemy 2.0
    *   MySQL & PostgreSQL support
    *   Docker
    *   Ruff (linter) and UV (package manager)
*   **Well-Defined Layers:**
    *   **API:**  Handles requests and responses (similar to Controller in MVC).
    *   **Schema:**  Defines data transfer objects (DTOs).
    *   **Service:**  Contains business logic.
    *   **CRUD:** Handles data access (similar to DAO/Mapper).
    *   **Model:**  Represents the data models.
*   **Easy Customization:**  Designed to be adaptable; modify and transform the architecture to fit your specific project needs.
*   **Comprehensive Documentation:**  Refer to the [official documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/) for in-depth details.

## Architecture Overview (Pseudo 3-Tier)

| Workflow        | Component               |
|-----------------|-------------------------|
| View            | API                     |
| Data Transfer   | Schema                  |
| Business Logic  | Service                 |
| Data Access     | CRUD                    |
| Model           | Model                   |

## Contributing

Explore the project's contributors:

<a href="https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=fastapi-practices/fastapi_best_architecture"/>
</a>

## Special Thanks

We'd like to give special thanks to the following technologies:

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)

## Get Involved

*   **Discord:**  [Discord](https://wu-clan.github.io/homepage/)

## Support the Project

If you find this project helpful, consider supporting us:

*   [:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

## License

This project is licensed under the [MIT License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE).

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)