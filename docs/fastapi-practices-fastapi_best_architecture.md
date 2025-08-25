<div align="center">

<img alt="The logo includes the abstract combination of the three letters FBA, forming a lightning bolt that seems to spread out from the ground" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">

# FastAPI Best Architecture: Build Robust Backend Solutions

This project offers a robust and scalable backend architecture solution built with FastAPI, providing a solid foundation for enterprise-level applications.  Discover more on the [original GitHub repository](https://github.com/fastapi-practices/fastapi_best_architecture).

English | [简体中文](./README.zh-CN.md)

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

</div>

## Key Features

*   **Modern Architecture:**  Employs a pseudo 3-tier architecture for a structured and maintainable codebase.
*   **FastAPI Framework:** Built upon FastAPI, leveraging its speed, performance, and ease of use.
*   **Data Validation:** Integrates Pydantic for robust data validation and schema definition.
*   **Database Support:** Supports MySQL, PostgreSQL, and SQLAlchemy for flexible database interactions.
*   **Code Quality:**  Uses Ruff for linting and code quality.
*   **Containerization:** Docker support for easy deployment and environment consistency.

## Architecture Overview: Pseudo 3-Tier

This project utilizes a "pseudo 3-tier" architecture, providing a clear separation of concerns, inspired by the principles of MVC, but tailored for FastAPI.

| Workflow       | Java           | FastAPI Best Architecture |
|----------------|----------------|---------------------------|
| View           | Controller     | API                       |
| Data Transfer  | DTO            | Schema                    |
| Business Logic | Service + Impl | Service                   |
| Data Access    | DAO / Mapper   | CRUD                      |
| Model          | Model / Entity | Model                     |

## Getting Started

Refer to the [official documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/) for detailed setup and usage instructions.

## Contributing

See the [Contributor's Guide](https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors) to learn how you can contribute to the project.

## Acknowledgements

Special thanks to the following projects:

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)
*   ...

## Community & Support

*   [Discord](https://wu-clan.github.io/homepage/)

## Sponsor

If this project has been helpful, consider supporting it with a coffee: [:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

## License

This project is licensed under the [MIT License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE).

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)