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

## Tracecat: Automate Security & IT Workflows with Open Source

Tracecat is a modern, open-source platform designed to automate security and IT response workflows.  [Explore the Tracecat repository](https://github.com/TracecatHQ/tracecat).

**Key Features:**

*   **YAML-Based Templates:**  Easily define integrations and workflows using simple YAML templates.
*   **No-Code UI:** A user-friendly interface for building and managing workflows without code.
*   **Built-in Lookup Tables:**  Store and reference data within your workflows.
*   **Case Management:**  Manage incidents and investigations directly within Tracecat.
*   **Scalable Orchestration:**  Powered by Temporal for robust, scalable, and reliable execution.
*   **Open Source:** Leverage the power of open-source automation.
*   **Integration Registry:** Quickly get started with community-contributed templates.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Full instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform.  View full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Community

Join the Tracecat community!  Ask questions, provide feedback, and share integration ideas in the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Tracecat Registry

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry is a central repository of pre-built integration and response-as-code templates.  These templates streamline common security and IT tasks.

*   **Ontology:** Response actions are organized using Tracecat's ontology of common capabilities (e.g., `list_alerts`, `list_cases`, `list_users`).
*   **OCSF Compliance:** Template inputs are normalized to the [Open Cyber Security Schema (OCSF)](https://schema.ocsf.io/) wherever possible.

**Examples & Templates:**

*   Find use cases and inspiration in the [Tracecat Registry documentation](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).
*   Explore open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, with the exception of the `ee` directory, which contains features requiring a Tracecat Enterprise license.

The Enterprise Edition provides advanced capabilities, supporting ongoing research and development efforts.

*For information on Tracecat's Enterprise self-hosted or managed cloud offerings, please visit [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).*

## Security

We are committed to security. SSO, audit logs, and IaC deployments (Terraform, Kubernetes / Helm) are always free and available. We're working on a comprehensive list of Tracecat's threat model, security features, and hardening recommendations. For immediate answers, reach out to us on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

A huge thank you to all our contributors! Open source thrives on your contributions. ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>