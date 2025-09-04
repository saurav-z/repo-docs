<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>

</br>

<div align="center">

[![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)](https://github.com/TracecatHQ/tracecat)
[![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)
[![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)

</div>

<div align="center">

<a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
<a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>

</div>

## Tracecat: Automate Security & IT Workflows with Open Source

Tracecat is an open-source workflow automation platform designed to streamline security and IT operations. ([Back to the top](https://github.com/TracecatHQ/tracecat))

**Key Features:**

*   **YAML-Based Templates:** Quickly build and deploy integrations with simple YAML configurations.
*   **No-Code UI:** Easily create and manage workflows through an intuitive user interface.
*   **Built-in Lookup Tables & Case Management:**  Manage and organize data effectively within your workflows.
*   **Scalable & Reliable:** Orchestrated using Temporal for robust performance and scalability.
*   **Open Cyber Security Schema (OCSF) Compliance:** Template inputs are normalized to fit the OCSF ontology where possible.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is in active development. Expect breaking changes with releases. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Find detailed instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry: Integration Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a library of pre-built templates to simplify your security and IT automation tasks. Response actions are organized based on a common ontology of capabilities (e.g., `list_alerts`, `list_cases`, `list_users`).

*   **Examples:**  Explore use cases and ideas within our documentation.
*   **Open Source Templates:**  Browse and use existing open-source templates in our [repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community to ask questions, share feedback, and collaborate on new integration ideas.  Connect with us on the [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, except for the `ee` directory, which contains features requiring a Tracecat Enterprise license. The Enterprise Edition offers advanced capabilities with dedicated research and development.  You can enable the Enterprise Edition directly within the platform's settings.

*   Interested in Tracecat's Enterprise self-hosted or managed cloud offerings?  Visit our [website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security.  SSO, audit logs, and infrastructure-as-code deployments (Terraform, Kubernetes/Helm) are free and available.  We are building a comprehensive threat model, security features, and hardening recommendations. For immediate answers, reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report any security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with `tracecat` in the subject line.

## Contributors

Thank you to all our contributors for their code, integrations, and support!

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">

  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>

</div>