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

## Tracecat: Automate Security and IT Workflows with Ease

Tracecat is an open-source automation platform designed to streamline security and IT response engineering.  [Explore the Tracecat repository](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-based Templates:** Create integrations with ease using simple YAML configurations.
*   **No-Code UI:** Build and manage workflows through an intuitive, no-code user interface.
*   **Built-in Lookup Tables & Case Management:** Simplify data management and incident tracking.
*   **Scalable Orchestration:** Powered by Temporal for reliability and scalability.
*   **Open Cyber Security Schema (OCSF) Compatibility:** Standardize your data inputs.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Find detailed instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community and connect with fellow users and developers! Share your ideas, ask questions, and stay up-to-date on the latest developments in the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry: Template Library

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a collection of integration and response-as-code templates.

**Key Benefits:**

*   **Organized Actions:** Response actions are categorized using a common ontology (e.g., `list_alerts`, `list_cases`, `list_users`).
*   **OCSF Alignment:** Template inputs are normalized to the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) when possible.

**Explore the Registry:**  Visit our documentation for use cases and ideas.  Explore the open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, excluding the `ee` directory, which contains features available with a Tracecat Enterprise license. The Enterprise Edition offers advanced features that require dedicated R&D investment. You can enable the Enterprise Edition directly in the platform settings.

*For information on Tracecat's Enterprise self-hosted or managed cloud offerings, please visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

Tracecat prioritizes security. We provide SSO, audit logs, and infrastructure-as-code deployments (Terraform, Kubernetes/Helm) free of charge. We're developing a comprehensive threat model, security features, and hardening recommendations.  For immediate security inquiries, contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security vulnerabilities to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

A huge thank you to all our contributors for their valuable contributions of code, integrations, and support! Your contributions make open source possible. ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>