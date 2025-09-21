<div align="center">
  <img src="img/banner.svg" alt="The workflow automation platform for security and IT response engineering.">
</div>
<br>

<div align="center">
  ![Commits](https://img.shields.io/github/commit-activity/m/TracecatHQ/tracecat?style=for-the-badge&logo=github)
  ![License](https://img.shields.io/badge/License-AGPL%203.0-blue?style=for-the-badge&logo=agpl)
  [![Discord](https://img.shields.io/discord/1212548097624903681.svg?style=for-the-badge&logo=discord&logoColor=white)](https://discord.gg/H4XZwsYzY4)
</div>

<div align="center">
  <a href="https://docs.tracecat.com"><img src="https://img.shields.io/badge/Documentation-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIyNCIgaGVpZ2h0PSIyNCIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMiIgc3Ryb2tlLWxpbmVjYXA9InJvdW5kIiBzdHJva2UtbGluZWpvaW49InJvdW5kIiBjbGFzcz0ibHVjaWRlIGx1Y2lkZS1ib29rLW9wZW4iPjxwYXRoIGQ9Ik0xMiA3djE0Ii8+PHBhdGggZD0iTTMgMThhMSAxIDAgMCAxLTEtMVY0YTEgMSAwIDAgMSAxLTFoNWE0IDQgMCAwIDEgNCA0IDQgNCAwIDAgMSA0LTRoNWExIDEgMCAwIDEgMSAxdjEzYTEgMSAwIDAgMS0xMWgtNmEzIDMgMCAwIDAtMyAzIDMgMCAwIDAtMy0zeiIvPjwvc3ZnPg==&logoColor=white"></a>
  <a href="https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates"><img src="https://img.shields.io/badge/Templates%20Library-%23000000.svg?style=for-the-badge&logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHdpZHRoPSIxNiIgaGVpZ2h0PSIxNiIgdmlld0JveD0iMCAwIDI0IDI0IiBmaWxsPSJub25lIiBzdHJva2U9IiNmZmZmZmYiIHN0cm9rZS13aWR0aD0iMS41IiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiIGNsYXNzPSJsdWNpZGUgbHVjaWRlLXNoaWVsZCI+PHBhdGggZD0iTTIwIDEzYzAgNS0zLjUgNy41LTcuNjYgOC45NWExIDEgMCAwIDEtLjY3LS4wMUM3LjUgMjAuNSA0IDE4IDQgMTNWNmExIDEgMCAwIDEgMS0xYzIgMCA0LjUtMS4yIDYuMjQtMi43MmExLjE3IDEuMTcgMCAwIDEgMS41MiAwQzE0LjUxIDMuODEgMTcgNSAxOSA1YTEgMSAwIDAgMSAxIDF6Ii8+PC9zdmc+&logoColor=white"></a>
</div>

## Tracecat: Automate Your Security and IT Workflows with Open Source Power

[Tracecat](https://github.com/TracecatHQ/tracecat) is a modern, open-source automation platform designed for security and IT engineers, offering a streamlined approach to incident response and workflow management.

**Key Features:**

*   **YAML-Based Templates:** Define integrations and workflows using simple, easy-to-understand YAML files.
*   **No-Code UI:**  A user-friendly interface for managing and executing your automated workflows.
*   **Built-in Lookup Tables & Case Management:** Organize and manage incident data efficiently.
*   **Orchestration with Temporal:** Benefit from scalable and reliable workflow execution powered by Temporal.
*   **Open Cyber Security Schema (OCSF) Compatibility**: Tracecat template inputs are normalized to fit the Open Cyber Security Schema (OCSF) ontology where possible.

![Tracecat workflow](/img/workflow.png)

## Getting Started

> [!IMPORTANT]
> Tracecat is under active development and breaking changes may occur. Review the release [changelog](https://github.com/TracecatHQ/tracecat/releases) before updating.

### Run Tracecat Locally

Deploy a local Tracecat stack using Docker Compose.  Find full instructions [here](https://docs.tracecat.com/self-hosting/deployment-options/docker-compose).

### Run Tracecat on AWS Fargate

**For advanced users:** Deploy a production-ready Tracecat stack on AWS Fargate using Terraform. Detailed instructions are available [here](https://docs.tracecat.com/self-hosting/deployment-options/aws-ecs).

### Run Tracecat on Kubernetes

Coming soon.

## Tracecat Registry: Integration and Response Templates

![Tracecat Action template](img/action-template.svg)

The Tracecat Registry provides a collection of pre-built integration and response templates, accelerating your automation efforts.  Response actions are categorized using Tracecat's ontology of common capabilities like `list_alerts`, `list_cases`, and `list_users`.

**Examples:**

*   Visit the documentation on the Tracecat Registry for use cases and ideas.
*   Explore existing open-source templates in [our repository](https://github.com/TracecatHQ/tracecat/tree/main/packages/tracecat-registry/tracecat_registry/templates).

## Community

Join the Tracecat community to ask questions, share feedback, and propose new integration ideas: [Tracecat Community Discord](https://discord.gg/H4XZwsYzY4).

## Open Source vs. Enterprise

This repository is licensed under AGPL-3.0, excluding the `ee` directory, which contains features available under a Tracecat Enterprise license. The Enterprise Edition offers advanced features that require dedicated development.

If you're interested in a self-hosted or managed Cloud offering, explore [our website](https://tracecat.com) or [book a meeting](https://cal.com/team/tracecat).

## Security

Tracecat prioritizes security, offering features such as SSO, audit logs, and Infrastructure as Code (IaaC) deployments (Terraform, Kubernetes / Helm).  We're developing a comprehensive security overview and hardening recommendations.  For immediate security inquiries, reach out on [Discord](https://discord.gg/H4XZwsYzY4).

Report security issues to [security@tracecat.com](mailto:founders+security@tracecat.com) with "tracecat" in the subject line.

## Contributors

A huge thank you to all our contributors! Open source thrives because of your contributions. ❤️

<a href="https://github.com/TracecatHQ/tracecat/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=TracecatHQ/tracecat" />
</a>

<br>
<br>

<div align="center">
  <sub>**`Tracecat`** is distributed under [**AGPL-3.0**](https://github.com/TracecatHQ/tracecat/blob/main/LICENSE)</sub>
</div>