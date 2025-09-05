<div align="center">
  <img alt="FastAPI Best Architecture Logo" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">
</div>

# FastAPI Best Architecture: Build Robust Backend Solutions

**Looking for a scalable and well-structured backend solution?** FastAPI Best Architecture provides a robust foundation for building enterprise-level applications with FastAPI. Explore the original repository on [GitHub](https://github.com/fastapi-practices/fastapi_best_architecture).

## Key Features

*   **Pseudo 3-Tier Architecture:** Designed with a clear separation of concerns, inspired by the 3-tier architectural pattern for enhanced maintainability and scalability.
*   **Modern Technologies:** Leverages cutting-edge technologies for performance and efficiency.
    *   Python 3.10+
    *   MySQL 8.0+
    *   PostgreSQL 16.0+
    *   SQLAlchemy 2.0
    *   Pydantic v2
    *   Ruff for linting and code formatting
    *   uv for fast package management
    *   Docker for containerization
*   **Data Validation & Serialization:** Utilizes Pydantic for robust data validation and serialization, ensuring data integrity.
*   **Comprehensive Documentation:** Detailed documentation to guide you through the architecture and its components.
*   **Community Support:** Engage with the community and get help via Discord.
*   **Open Source:** Released under the MIT License, allowing for flexible use and modification.

## Architecture Overview

This project implements a "pseudo 3-tier" architecture, mapping concepts from other frameworks like SpringBoot to FastAPI:

| Workflow       | FastAPI Best Architecture | Equivalent Concepts                 |
|----------------|---------------------------|-------------------------------------|
| View           | API                       | Controller                          |
| Data Transfer  | Schema                    | DTO (Data Transfer Object)            |
| Business Logic | Service                   | Service + Impl                      |
| Data Access    | CRUD                      | DAO / Mapper                       |
| Model          | Model                     | Model / Entity                      |

## Get Started

*   **Documentation:** Detailed information is available in the [official documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/).
*   **Discord:** Join the community on [Discord](https://discord.com/invite/yNN3wTbVAC) for support and discussions.

## Special Thanks

We would like to thank the following projects and contributors:

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)
*   ...

## Contributors

<a href="https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=fastapi-practices/fastapi_best_architecture"/>
</a>

## Support the Project

If you find this project helpful, consider supporting the developers:
[:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

## License

This project is licensed under the [MIT](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE) license.

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)