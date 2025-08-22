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

## Tracecat: Automate Security and IT Workflows with Open Source

[Tracecat](https://github.com/TracecatHQ/tracecat) is a modern, open-source platform designed to empower security and IT engineers by automating workflows.

**Key Features:**

*   **YAML-Based Templates:** Define integrations with simple, easy-to-understand YAML templates.
*   **No-Code UI:**  Build and manage workflows effortlessly with an intuitive user interface.
*   **Built-in Lookup Tables & Case Management:** Streamline investigations and response with integrated features.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust and scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Compliance:** Template inputs normalize to OCSF for consistent data.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is under active development.  Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating due to potential breaking changes.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Find comprehensive instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon!

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

Tracecat Registry provides a library of pre-built integration and response-as-code templates. These templates are organized using a common ontology for efficient use.

**Key Benefits:**

*   **Pre-built Integrations:**  Leverage ready-to-use templates for common security and IT tasks.
*   **Standardized Actions:** Actions are categorized based on their capabilities (e.g., `list_alerts`, `list_cases`).
*   **OCSF Alignment:** Template inputs are designed to align with the Open Cyber Security Schema (OCSF).

**Explore Templates:**

*   Visit our [documentation](https://docs.tracecat.com) for use cases and ideas.
*   Browse the open-source templates in the [Tracecat Registry repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community!  Ask questions, provide feedback, and share integration ideas on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, with the exception of the `ee` directory, which contains enterprise features requiring a Tracecat Enterprise license. Enterprise features provide additional capabilities.

For more information about Tracecat's Enterprise self-hosted or managed Cloud offering, visit the [Tracecat website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security.  Key features such as SSO, audit logs, and IaaC deployments (Terraform, Kubernetes/Helm) are freely available. We're actively developing a comprehensive threat model and security features documentation. For urgent security inquiries, contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including `tracecat` in the subject line.

## Contributors

Thank you to our amazing contributors!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>