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

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xIDFoLTZhMyAzIDAgMCAwLTMgMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

# Tracecat: Automate Security and IT Workflows with Open Source Power

Tracecat is a modern, open-source platform designed to streamline security and IT response engineering, empowering you to automate workflows with ease.  **[Check out the original repo](https://github.com/TracecatHQ/tracecat).**

## Key Features

*   **YAML-Based Templates:** Define integrations and workflows using simple, YAML-based templates for easy customization.
*   **No-Code UI:** Leverage an intuitive no-code UI to build and manage workflows without complex coding.
*   **Built-in Lookup Tables & Case Management:** Enhance automation with built-in lookup tables and case management capabilities.
*   **Scalable Orchestration:** Built on Temporal for robust scale and reliability, ensuring your workflows run smoothly.
*   **Open Cyber Security Schema (OCSF) Compatibility:**  Template inputs are normalized to fit the OCSF ontology where possible.

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  For advanced users, find full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

Tracecat Registry provides a collection of integration and response-as-code templates to accelerate your automation efforts.

*   **Action Ontology:** Response actions are organized within Tracecat's own ontology of common capabilities (e.g., `list_alerts`, `list_cases`, `list_users`).
*   **Template Library:** Explore use cases and find inspiration in our documentation on Tracecat Registry.
*   **Open Source Templates:** Explore open-source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, excluding the `ee` directory, which contains paid enterprise features requiring a Tracecat Enterprise license.  The Enterprise Edition provides advanced features and functionalities.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offerings, please visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Tracecat prioritizes security, offering features like SSO, audit logs, and IaC deployments (Terraform, Kubernetes/Helm).  We are working on comprehensive documentation of Tracecat's threat model, security features, and hardening recommendations.  For immediate security-related questions, reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Community & Support

Join the Tracecat community!  Share your questions, feedback, and integration ideas on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Contributors

Thank you to all the amazing contributors who are making open source a reality!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>