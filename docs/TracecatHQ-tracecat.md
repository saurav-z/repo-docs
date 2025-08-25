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

# Tracecat: Automate Security and IT Workflows with Ease

Tracecat is a modern, open-source workflow automation platform designed to empower security and IT engineers to streamline incident response and operational tasks. [Explore the Tracecat Repository](https://github.com/TracecatHQ/tracecat).

## Key Features

*   **YAML-Based Templates:** Easily define integrations and workflows using simple, human-readable YAML configurations.
*   **No-Code UI:**  Build and manage workflows visually with an intuitive user interface, eliminating the need for complex coding.
*   **Built-in Lookup Tables & Case Management:**  Organize and manage data efficiently with integrated lookup tables and case management capabilities.
*   **Scalable Orchestration:** Powered by Temporal for reliable, scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Compatibility:** Templates leverage and adhere to the OCSF standard for data normalization.

## Getting Started

> [!IMPORTANT]
> Tracecat is actively developed, and breaking changes may occur. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Comprehensive instructions can be found [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry: Integration & Response-as-Code Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry offers a curated collection of integration and response-as-code templates to accelerate your automation efforts. Response actions are structured using Tracecat's ontology of common capabilities (e.g., `list_alerts`, `list_cases`, `list_users`), and template inputs are normalized against the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) where possible.

**Examples:**

*   [Visit Tracecat Registry documentation for use cases and ideas.](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose)
*   [Browse existing open-source templates](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community to ask questions, provide feedback, and share integration ideas! Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, with the exception of the `ee` directory, which contains features requiring a Tracecat Enterprise license. The Enterprise Edition provides advanced features that are the result of specific investments in research and development.  You can enable the Enterprise Edition directly in the platform settings.

*For information on Tracecat's Enterprise self-hosted or managed Cloud offering, visit [our website](https://tracecat.com) or [book a meeting with us](https://cal.com/team/tracecat).*

## Security

SSO, audit logs, and IaaC deployments (Terraform, Kubernetes / Helm) will always be free and available. We're working on a comprehensive list of Tracecat's threat model, security features, and hardening recommendations.  For immediate answers to these questions, please reach to us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com), including `tracecat` in the subject line.

## Contributors

Thank you to our amazing contributors for your valuable contributions!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>