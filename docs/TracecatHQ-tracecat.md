<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

</br>

<div align="center">

![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)
![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)
[![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)

</div>

<div align="center">

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

# Tracecat: Automate Security and IT Workflows with Open Source

Tracecat is a modern, open-source platform that empowers security and IT engineers to automate critical workflows.  ([See the original repo](https://github.com/TracecatHQ/tracecat))

## Key Features

*   **YAML-Based Templates:** Create integrations and workflows using simple, easy-to-understand YAML templates.
*   **No-Code UI:**  Build and manage workflows with an intuitive, no-code user interface.
*   **Built-in Lookup Tables & Case Management:** Organize and track information efficiently.
*   **Orchestration with Temporal:** Benefit from robust, scalable, and reliable workflow execution.
*   **Open Cyber Security Schema (OCSF) Integration:**  Templates leverage the OCSF ontology where possible.

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Comprehensive instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon!

## Tracecat Registry:  Integration and Response-as-Code Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a collection of pre-built integration and response-as-code templates. Response actions are organized using Tracecat's ontology of common capabilities (e.g., `list_alerts`, `list_cases`, `list_users`).

**Key Benefits:**

*   **Pre-built Actions:**  Quickly implement common security and IT tasks.
*   **Standardized Inputs:** Templates utilize normalized inputs aligned with the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) for consistency.

**Explore the Registry:**

*   Visit the [Tracecat Registry Documentation](https://docs.tracecat.com) for use cases and ideas.
*   Browse the open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community for support, feedback, and to share new integration ideas!  Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, excluding the `ee` directory, which contains features available with a Tracecat Enterprise license.  The Enterprise Edition offers advanced features, and to learn more or book a meeting, check out [Tracecat's website](https://tracecat.com).

## Security

Tracecat prioritizes security with features like SSO, audit logs, and Infrastructure-as-Code deployments.  For security inquiries, contact [security@tracecat.com](mailto:founders+security@tracecat.com) and include `tracecat` in the subject line.  For immediate answers, reach out on our [Discord](https://discord.gg/H4XZwsYzY4).

## Contributors

A huge thank you to all our contributors! ❤️ Open source is only possible because of you.

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>