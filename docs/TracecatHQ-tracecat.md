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

**Tracecat is an open-source workflow automation platform that empowers security and IT engineers to streamline their incident response and operational tasks.**

[![Tracecat Workflow](img/workflow.png)](https://github.com/TracecatHQ/tracecat)

## Key Features

*   **YAML-Based Templates:** Create and customize workflows easily with simple YAML-based templates.
*   **No-Code UI:** Utilize a user-friendly interface for workflow management and orchestration.
*   **Built-in Lookup Tables & Case Management:** Manage and organize information within the platform.
*   **Scalable & Reliable Orchestration:** Powered by Temporal for robust and scalable workflow execution.
*   **Open Cyber Security Schema (OCSF) Compliance:** Templates are normalized to fit the OCSF ontology where possible.
*   **Extensive Template Library:** Leverage a growing collection of pre-built templates for common security and IT tasks.

## Getting Started

**Important:** Tracecat is under active development. Review the [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose. Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community for questions, feedback, and new integration ideas! Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action Template](img/action-template.svg)

The Tracecat Registry is a comprehensive library of integration and response-as-code templates.  These actions are categorized according to Tracecat's ontology of common capabilities like `list_alerts` and `list_cases`.

**Examples:**

*   Visit our documentation on Tracecat Registry for use cases and ideas.
*   Explore existing open-source templates in our [repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is available under the AGPL-3.0 license, excluding the `ee` directory, which contains features available with a Tracecat Enterprise license. The Enterprise Edition offers advanced features that require dedicated research and development.  You can enable the Enterprise Edition directly in the platform settings.

For information on Tracecat's Enterprise self-hosted or managed Cloud offerings, please visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

SSO, audit logs, and IaC deployments (Terraform, Kubernetes/Helm) are always free.  We are developing a detailed threat model and security features list.  For immediate security inquiries, please contact us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

A big thank you to all our contributors!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>