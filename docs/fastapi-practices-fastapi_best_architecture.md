<div align="center">

<img alt="The logo includes the abstract combination of the three letters FBA, forming a lightning bolt that seems to spread out from the ground" width="320" src="https://wu-clan.github.io/picx-images-hosting/logo/fba.png">

# FastAPI Best Architecture: Build Scalable Backend Applications

This project offers a robust, enterprise-ready architectural solution for building scalable and maintainable backend applications with FastAPI.  Check out the [original repository](https://github.com/fastapi-practices/fastapi_best_architecture) for more details!

</div>

## Key Features

*   **Pseudo 3-Tier Architecture:** Designed with a flexible, pseudo 3-tier architecture inspired by common patterns.  This structure promotes code organization and maintainability.
    *   **API (View):**  Handles incoming requests and returns responses.
    *   **Schema (Data Transfer):** Defines data structures for request and response payloads using Pydantic.
    *   **Service (Business Logic):** Encapsulates the core business logic of your application.
    *   **CRUD (Data Access):** Manages interactions with the database using SQLAlchemy.
    *   **Model:** Represents the data entities in your application.
*   **Technology Stack:** Leverages industry-leading technologies for a modern backend:
    *   FastAPI
    *   SQLAlchemy
    *   Pydantic v2
    *   MySQL, PostgreSQL
    *   Docker
    *   Ruff (Linter)
    *   uv (Package Manager)

## Architecture Overview

This project utilizes a "pseudo 3-tier" architecture, offering a clear separation of concerns:

| Component          | Description                                                        |
| ------------------ | ------------------------------------------------------------------ |
| API (View)         | Handles HTTP requests and responses.                              |
| Schema             | Defines data structures using Pydantic for data transfer.        |
| Service            | Contains the application's business logic.                         |
| CRUD               | Manages data access and interactions with the database.           |
| Model              | Represents the data entities, often corresponding to database tables. |

## Resources

*   [Official Documentation](https://fastapi-practices.github.io/fastapi_best_architecture_docs/)
*   [License](https://github.com/fastapi-practices/fastapi_best_architecture/blob/master/LICENSE)

## Acknowledgements

*   [FastAPI](https://fastapi.tiangolo.com/)
*   [Pydantic](https://docs.pydantic.dev/latest/)
*   [SQLAlchemy](https://docs.sqlalchemy.org/en/20/)
*   [Casbin](https://casbin.org/zh/)
*   [Ruff](https://beta.ruff.rs/docs/)

## Community & Support

*   [Discord](https://discord.com/invite/yNN3wTbVAC)
*   [Ask DeepWiki](https://deepwiki.com/fastapi-practices/fastapi_best_architecture)

## Contributing

[View Contributors](https://github.com/fastapi-practices/fastapi_best_architecture/graphs/contributors)

## Sponsor

[:coffee: Sponsor :coffee:](https://wu-clan.github.io/sponsor/)

[![Stargazers over time](https://starchart.cc/fastapi-practices/fastapi_best_architecture.svg?variant=adaptive)](https://starchart.cc/fastapi-practices/fastapi_best_architecture)