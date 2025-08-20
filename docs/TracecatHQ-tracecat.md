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

# Tracecat: Automate Security and IT Workflows with Open Source

Tracecat is a powerful, open-source automation platform designed to streamline security and IT response engineering. ([View on GitHub](https://github.com/TracecatHQ/tracecat))

## Key Features

*   **YAML-Based Templates:** Define integrations and workflows using simple, easy-to-understand YAML templates.
*   **No-Code UI:**  Easily build and manage workflows with an intuitive no-code user interface.
*   **Built-in Lookup Tables & Case Management:** Simplify data management and incident response with integrated lookup tables and case management capabilities.
*   **Scalable Orchestration:** Built on Temporal for reliable and scalable workflow execution.
*   **Tracecat Registry:** A collection of pre-built templates for common security and IT tasks, organized by a standardized ontology.

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate (Advanced)

Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  See the instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry offers a library of integration and response-as-code templates.  Response actions are structured according to Tracecat's ontology, making them easy to use and integrate.  Template inputs are normalized using the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) when possible.

**Examples and Resources:**

*   Explore use cases and ideas in our documentation.
*   Browse the open-source templates in [our repo](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community to ask questions, share feedback, and discuss new integration ideas on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, excluding the `ee` directory, which contains features available with a Tracecat Enterprise license. The Enterprise Edition provides additional, advanced features requiring specific development efforts.  You can enable the Enterprise Edition within the platform's settings.

For information on Tracecat's Enterprise self-hosted or managed cloud offerings, check out the [Tracecat website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

SSO, audit logs, and IaC deployments (Terraform, Kubernetes/Helm) are free and open-source.  We are actively developing a threat model, security features, and hardening recommendations.  For immediate security questions, please contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

A huge thank you to all our contributors for their code, integrations, and support!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>