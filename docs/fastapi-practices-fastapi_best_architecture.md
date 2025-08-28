<div align="center">
  <img alt="FastAPI Best Architecture Logo" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">
</div>

# FastAPI Best Architecture: Build Robust Backend Systems

**Looking for a production-ready architecture for your FastAPI projects?** This repository provides a comprehensive and scalable backend architecture solution built with FastAPI, designed for enterprise-level applications.

[**View the original repository on GitHub**](https://github.com/fastapi-practices/fastapi_best_architecture)

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

*   **Robust Architecture:**  Implements a pseudo 3-tier architecture for clear separation of concerns and maintainability.
*   **Modern Technologies:** Leverages FastAPI, Pydantic, SQLAlchemy, and other cutting-edge tools.
*   **Database Support:** Compatible with both MySQL and PostgreSQL.
*   **Scalability:** Designed for building enterprise-level applications.
*   **Code Quality:** Enforces code style and quality with Ruff and uv.
*   **Dockerization:** Includes Docker support for easy deployment and containerization.
*   **Comprehensive Documentation:** Detailed [official documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/) is available.

## Architecture Overview

This project adopts a "pseudo 3-tier" architecture approach, offering a structured way to organize your FastAPI application. Here's how it maps to common concepts:

| Component       | Purpose                             |
|----------------|-------------------------------------|
| API            | Handles incoming requests (like Controller) |
| Schema         | Defines data transfer objects (DTOs) |
| Service        | Implements business logic (like Service + Impl) |
| CRUD           | Manages data access (like DAO / Mapper)   |
| Model          | Represents data entities                  |

## Contributing

We welcome contributions!  See the project's [contributor graphs](https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors) to see who's already involved.

## Acknowledgements

We extend our gratitude to the following projects and their communities:

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)
*   ... and other contributors!

## Get Involved

*   [Join the Discord](https://wu-clan.github.io/homepage/)

## Support the Project

If you find this project helpful, consider supporting us with a coffee!  [:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

## License

This project is licensed under the [MIT License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE).

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)